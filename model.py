from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
import logging
import coref_cpp

logger = logging.getLogger(__name__)


class CorefModel(nn.Module, ABC):
    def __init__(self, config):
        super(CorefModel, self).__init__()
        self.config = config
        self.genres = {g: i for i, g in enumerate(config["genres"])}

        self.bert = BertModel.from_pretrained(config['init_checkpoint_dir'])

        self.span_width_embeddings = nn.Embedding(config['max_span_width'], config['feature_size'])
        nn.init.trunc_normal_(self.span_width_embeddings.weight.data, std=0.02)

        self.span_width_prior_embeddings = nn.Embedding(config['max_span_width'], config['feature_size'])
        nn.init.trunc_normal_(self.span_width_prior_embeddings.weight.data, std=0.02)

        self.genre_embeddings = nn.Embedding(len(self.genres), config['feature_size'])
        nn.init.trunc_normal_(self.genre_embeddings.weight.data, std=0.02)

        self.antecedent_distance_embeddings = nn.Embedding(10, config['feature_size'])
        nn.init.trunc_normal_(self.antecedent_distance_embeddings.weight.data, std=0.02)

        self.antecedent_distance_embeddings_coref_layer = nn.Embedding(10, config['feature_size'])
        nn.init.trunc_normal_(self.antecedent_distance_embeddings.weight.data, std=0.02)

        self.same_speaker_embeddings = nn.Embedding(2, config['feature_size'])
        nn.init.trunc_normal_(self.same_speaker_embeddings.weight.data, std=0.02)

        self.segment_distance_embeddings = nn.Embedding(config['max_training_sentences'], config['feature_size'])
        nn.init.trunc_normal_(self.segment_distance_embeddings.weight.data, std=0.02)

        self.dropout_rate = config['dropout_rate']
        span_embedding_size = 3 * config['hidden_size'] + config['feature_size']

        self.mention_word_attn_layer = self.ffnn(input_size=config['hidden_size'],
                                                 num_hidden_layers=0,
                                                 hidden_size=-1,
                                                 output_size=1,
                                                 dropout=None)

        self.mention_score_layer = self.ffnn(input_size=span_embedding_size,
                                             num_hidden_layers=config['ffnn_depth'],
                                             hidden_size=config['ffnn_size'],
                                             output_size=1,
                                             dropout=self.dropout_rate)

        self.width_scores_layer = self.ffnn(input_size=config['feature_size'],
                                            num_hidden_layers=config['ffnn_depth'],
                                            hidden_size=config['ffnn_size'],
                                            output_size=1,
                                            dropout=self.dropout_rate)
        self.src_projection = self.ffnn(input_size=span_embedding_size,
                                        num_hidden_layers=0,
                                        hidden_size=-1,
                                        output_size=span_embedding_size,
                                        dropout=None)

        self.antecedent_distance_scores_projection = self.ffnn(input_size=config['feature_size'],
                                                               num_hidden_layers=0,
                                                               hidden_size=-1,
                                                               output_size=1,
                                                               dropout=None)

        meta_feature_size = 4 * config['feature_size'] if config['use_metadata'] else 3 * config['feature_size']
        self.slow_antecedent_scores_layer = self.ffnn(
            input_size=3 * span_embedding_size + meta_feature_size,
            num_hidden_layers=config["ffnn_depth"],
            hidden_size=config["ffnn_size"],
            output_size=1,
            dropout=self.dropout_rate)

    def forward(self, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids,
                sentence_map, subtoken_map):
        """
            forward
            Args:
                input_ids: [num_sentences, max_segment_len]
                input_mask: [num_sentences, max_segment_len]
                text_len: [num_sentences]
                speaker_ids: [num_sentences, max_segment_len]
                genre: [1]
                gold_starts: [num_mentions]
                gold_ends: [num_mentions]
                cluster_ids: [num_mentions]
                sentence_map: [num_tokens]
                subtoken_map: [num_tokens]

            Returns:

        """
        num_tokens = sentence_map.size()[0]
        # [num_tokens, embed_size]
        token_embeddings, flatten_input_ids = self.get_bert_embeddings(input_ids, input_mask)

        # [num_candidates]
        candidate_starts, candidate_ends, candidate_mask = self.get_candidate_spans(sentence_map)

        # [num_candidates]
        candidate_cluster_ids = self.get_candidate_labels(candidate_starts=candidate_starts,
                                                          candidate_ends=candidate_ends,
                                                          labeled_starts=gold_starts,
                                                          labeled_ends=gold_ends,
                                                          labels=cluster_ids)

        # [num_candidates, 3 * embed_size + feature_size]
        candidate_span_emb = self.get_span_embeddings(token_embeddings=token_embeddings,
                                                      span_starts=candidate_starts,
                                                      span_ends=candidate_ends)

        candidate_mention_scores = self.get_mention_scores(span_emb=candidate_span_emb,
                                                           span_starts=candidate_starts,
                                                           span_ends=candidate_ends)
        mention_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            candidate_mention_scores,
            (candidate_cluster_ids > 0).float()
        )

        num_candidates = candidate_mention_scores.size(0)
        k = min(int(num_tokens * self.config["top_span_ratio"]), self.config["max_num_candidates"])

        c = min(self.config["max_top_antecedents"], k)
        sorted_span_mention_scores, sorted_span_indices = torch.topk(candidate_mention_scores, num_candidates)
        top_span_indices = coref_cpp.extract_spans(sorted_span_indices.cpu(), candidate_starts.cpu(),
                                                   candidate_ends.cpu(), k, num_tokens, True)
        top_span_mention_scores = candidate_mention_scores[top_span_indices]
        top_span_starts = candidate_starts[top_span_indices]
        top_span_ends = candidate_ends[top_span_indices]
        top_span_emb = candidate_span_emb[top_span_indices]
        top_span_cluster_ids = candidate_cluster_ids[top_span_indices]

        # [1, feature_size]
        genre_emb = self.genre_embeddings(genre)

        if self.config['use_metadata']:
            # [num_sentences * max_segment_len]
            speaker_ids = torch.reshape(speaker_ids, [-1])
            flatten_mask = torch.reshape(input_mask, [-1])
            speaker_ids = speaker_ids[flatten_mask > 0]
            top_span_speaker_ids = speaker_ids[top_span_starts]
        else:
            top_span_speaker_ids = None

        dummy_scores = torch.zeros([k, 1]).to(input_ids.device)
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets \
            = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c, top_span_cluster_ids)

        num_sentences, max_segment_len = input_ids.size()
        word_segments = torch.arange(0, num_sentences).unsqueeze(1).expand([-1, max_segment_len]).to(input_ids.device)
        # [num_tokens]
        flatten_word_segments = word_segments.reshape([-1])[input_mask.reshape([-1]) > 0]
        # [k, 1]
        mention_segments = flatten_word_segments[top_span_starts].unsqueeze(1)
        # [k, c]
        antecedent_segments = flatten_word_segments[top_span_starts[top_antecedents]]
        # [k, c]
        segment_distance = torch.clamp(mention_segments - antecedent_segments, min=0,
                                       max=self.config['max_training_sentences'] - 1) if self.config[
            'use_segment_distance'] else None

        top_antecedent_emb = top_span_emb[top_antecedents]
        top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                             top_antecedents,
                                                                                             top_antecedent_emb,
                                                                                             top_antecedent_offsets,
                                                                                             genre_emb,
                                                                                             top_span_speaker_ids,
                                                                                             segment_distance)

        # [k, c + 1]
        top_antecedent_scores = torch.cat([dummy_scores, top_antecedent_scores], 1)
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedents]
        top_antecedent_cluster_ids += torch.log(top_antecedents_mask.float()).long()
        same_cluster_indicator = top_antecedent_cluster_ids == top_span_cluster_ids.unsqueeze(1)
        non_dummy_indicator = (top_span_cluster_ids > 0).unsqueeze(1)
        pairwise_labels = torch.logical_and(same_cluster_indicator, non_dummy_indicator)
        dummy_labels = torch.logical_not(torch.any(pairwise_labels, dim=1, keepdims=True))
        top_antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1).long()
        coref_loss = self.marginal_likelihood(top_antecedent_scores, top_antecedent_labels)
        loss = coref_loss + self.config['mention_loss_ratio'] * mention_loss

        return [top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, candidate_starts,
                candidate_ends, top_span_cluster_ids, top_span_mention_scores, candidate_mention_scores], loss

    def marginal_likelihood(self, antecedent_scores, antecedent_labels):
        """
        Desc:
            marginal likelihood of gold antecedent spans from coreference clusters.
        Args:
            antecedent_scores: [k, c+1] the predicted scores by the model
            antecedent_labels: [k, c+1] the gold-truth cluster labels
        Returns:
            a scalar of loss
        """
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())
        marginalized_gold_scores = torch.logsumexp(gold_scores, 1)  # [k]
        log_norm = torch.logsumexp(antecedent_scores, 1)  # [k]
        loss = log_norm - marginalized_gold_scores  # [k]
        return loss.sum()

    def mask_logits(self, target, mask):
        return target * mask + (1 - mask) * (-10)

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c, top_span_cluster_ids):
        k = top_span_emb.size()[0]
        top_span_range = torch.arange(0, k).to(top_span_emb.device)
        # [k, k]
        antecedent_offsets = top_span_range.unsqueeze(1) - top_span_range.unsqueeze(0)
        antecedents_mask = antecedent_offsets >= 1
        # [k, k]
        fast_antecedent_scores = self.get_fast_antecedent_scores(top_span_emb)
        fast_antecedent_scores += top_span_mention_scores.unsqueeze(1) + top_span_mention_scores.unsqueeze(0)
        fast_antecedent_scores += torch.log(antecedents_mask.float())
        if self.config['use_prior']:
            antecedent_distance_buckets = self.bucket_distance(antecedent_offsets)  # [k, c]
            antecedent_distance_emb = self.antecedent_distance_embeddings(antecedent_distance_buckets)
            antecedent_distance_emb = torch.dropout(antecedent_distance_emb, p=self.dropout_rate, train=self.training)
            antecedent_distance_scores = self.antecedent_distance_scores_projection(antecedent_distance_emb).squeeze(2)
            fast_antecedent_scores += antecedent_distance_scores
        # [k, c]
        _, top_antecedents = torch.topk(fast_antecedent_scores, c, sorted=False)

        top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents)
        top_fast_antecedent_scores = self.batch_gather(fast_antecedent_scores, top_antecedents)
        top_antecedent_offsets = self.batch_gather(antecedent_offsets, top_antecedents)
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   genre_emb, top_span_speaker_ids, segment_distance=None):
        device = top_span_emb.device
        k = top_span_emb.size()[0]
        c = top_antecedents.size()[1]
        feature_emb_list = []
        if self.config["use_metadata"]:
            # [k, c]
            top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedents]
            same_speaker = (top_span_speaker_ids.unsqueeze(1) == top_antecedent_speaker_ids).long().to(device)
            speaker_pair_emb = self.same_speaker_embeddings(same_speaker)
            feature_emb_list.append(speaker_pair_emb)
            # [k, c, feature_size]
            tiled_genre_emb = genre_emb.unsqueeze(0).expand([k, c, -1])
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)
            antecedent_distance_emb = self.antecedent_distance_embeddings_coref_layer(antecedent_distance_buckets)
            feature_emb_list.append(antecedent_distance_emb)

        if segment_distance is not None:
            # [k, c, feature_size]
            segment_distance_emb = self.segment_distance_embeddings(segment_distance)
            feature_emb_list.append(segment_distance_emb)

        # [k, c, emb]
        feature_emb = torch.dropout(torch.cat(feature_emb_list, 2), p=self.dropout_rate, train=self.training)
        # [k, 1, emb]
        target_emb = top_span_emb.unsqueeze(1)
        # [k, c, emb]
        similarity_emb = top_antecedent_emb * target_emb
        target_emb = target_emb.expand([-1, c, -1])
        pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)

        slow_antecedent_scores = self.slow_antecedent_scores_layer(pair_emb).squeeze(2)
        return slow_antecedent_scores

    def batch_gather(self, emb, indices):
        batch_size = emb.size()[0]
        seqlen = emb.size()[1]
        if len(emb.size()) > 2:
            emb_size = emb.size()[2]
        else:
            emb_size = 1
        flattened_emb = torch.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
        offset = (torch.arange(0, batch_size) * seqlen).unsqueeze(1).to(indices.device)  # [batch_size, 1]
        gathered = flattened_emb[indices + offset]  # [batch_size, num_indices, emb]
        if len(emb.size()) == 2:
            gathered = gathered.squeeze(2)  # [batch_size, num_indices]
        return gathered

    def bucket_distance(self, distances):
        """
            Places the given values (designed for distances) into 10 semi-logscale buckets:
            [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = torch.floor(
            torch.log(distances.float()) / torch.log(torch.ones([1]).fill_(2)).to(distances.device)).long() + 3
        use_identity = (distances <= 4).long()
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, min=0, max=9)

    def get_fast_antecedent_scores(self, top_span_emb):
        source_top_span_emb = torch.dropout(self.src_projection(top_span_emb), p=self.dropout_rate, train=self.training)
        target_top_span_emb = torch.dropout(top_span_emb, p=self.dropout_rate, train=self.training)
        # [k, k]
        return torch.matmul(source_top_span_emb, target_top_span_emb.t())

    def get_bert_embeddings(self, input_ids, input_mask):
        # [max_training_sentences, max_segment_len, embed_size]
        outputs = self.bert(input_ids, (input_mask != 0).long())
        bert_embeddings = outputs[0]
        num_sentences, max_segment_len = input_ids.size()
        flatten_embeddings = torch.reshape(bert_embeddings, [num_sentences * max_segment_len, -1])
        flatten_mask = torch.reshape(input_mask > 0, [num_sentences * max_segment_len])
        flatten_input_ids = torch.reshape(input_ids, [num_sentences * max_segment_len])
        return flatten_embeddings[flatten_mask], flatten_input_ids[flatten_mask]

    def get_candidate_spans(self, sentence_map):
        num_tokens = sentence_map.size()[0]

        # [num_tokens, max_span_width]
        candidate_starts = torch.arange(0, num_tokens, dtype=torch.int64).unsqueeze(1).expand(-1, self.config[
            'max_span_width']).contiguous()
        # [num_tokens, max_span_width]
        candidate_ends = candidate_starts + torch.arange(0, self.config['max_span_width'], dtype=torch.int64). \
            unsqueeze(0).expand(num_tokens, -1).contiguous()
        # [num_tokens * max_span_width]
        candidate_starts = candidate_starts.view(-1).to(sentence_map.device)
        candidate_ends = candidate_ends.view(-1).to(sentence_map.device)
        # [num_tokens * max_span_width]
        candidate_start_sentence_indices = sentence_map[candidate_starts]
        candidate_end_sentence_indices = sentence_map[torch.clamp(candidate_ends, min=0, max=num_tokens - 1)]
        candidate_mask = torch.logical_and(
            candidate_ends < num_tokens,
            (candidate_start_sentence_indices - candidate_end_sentence_indices) == 0,
        )

        # [num_candidates]
        candidate_starts = candidate_starts[candidate_mask]
        candidate_ends = candidate_ends[candidate_mask]

        return candidate_starts, candidate_ends, candidate_mask

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels=None):
        """
        Args:
            candidate_starts: [num_candidates]
            candidate_ends: [num_candidates]
            labeled_starts: [num_mentions]
            labeled_ends: [num_mentions]
            labels: [num_mentions]

        Returns:
            candidate_labels: [num_candidates]
        """
        # [num_mentions, num_candidates]
        same_start = labeled_starts.unsqueeze(1) == candidate_starts.unsqueeze(0)
        same_end = labeled_ends.unsqueeze(1) == candidate_ends.unsqueeze(0)
        same_span = torch.logical_and(same_start, same_end)

        # [1, num_mentions] * [num_mentions, num_candidates] -> [1, num_candidates]
        labels = labels.unsqueeze(0).float()
        candidate_labels = torch.matmul(labels, same_span.float())
        candidate_labels = candidate_labels.long().squeeze(0)
        return candidate_labels

    def get_span_embeddings(self, token_embeddings, span_starts, span_ends):
        span_emb_list = []
        span_start_emb = token_embeddings[span_starts]  # [num_candidates, embed_size]
        span_end_emb = token_embeddings[span_ends]  # [num_candidates, embed_size]
        span_emb_list.append(span_start_emb)
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [num_candidates]

        if self.config["use_features"]:
            span_width_index = span_width - 1
            span_width_emb = self.span_width_embeddings(span_width_index)
            span_width_emb = torch.dropout(span_width_emb, self.dropout_rate, train=self.training)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(token_embeddings, span_starts, span_ends)
            # [num_candidates, num_tokens] * [num_tokens, embed_size] -> [num_candidates, embed_size]
            head_attn_reps = torch.matmul(mention_word_scores, token_embeddings)
            span_emb_list.append(head_attn_reps)

        # [num_candidates, 3 * embed_size + feature_size]
        span_emb = torch.cat(span_emb_list, dim=-1)
        return span_emb

    def get_masked_mention_word_scores(self, token_embeddings, span_starts, span_ends):
        num_tokens = token_embeddings.size()[0]
        num_candidates = span_starts.size()[0]

        doc_range = torch.arange(0, num_tokens, device=token_embeddings.device).unsqueeze(0).expand(num_candidates, -1)
        # [num_candidates, num_tokens]
        mention_mask = torch.logical_and(
            doc_range >= span_starts.unsqueeze(1),
            doc_range <= span_ends.unsqueeze(1),
        )
        # [num_tokens]
        word_atten = self.mention_word_attn_layer(token_embeddings).squeeze(1)
        # [num_candidates, num_tokens]
        mention_word_attn = torch.softmax((torch.log(mention_mask.float()) + word_atten.unsqueeze(0)), dim=-1)
        del mention_mask
        return mention_word_attn

    def get_mention_scores(self, span_emb, span_starts, span_ends):
        # [num_candidates]
        span_scores = self.mention_score_layer(span_emb).squeeze(1)
        if self.config['use_prior']:
            span_width_index = span_ends - span_starts
            span_width_emb = self.span_width_prior_embeddings(span_width_index)
            width_scores = self.width_scores_layer(span_width_emb).squeeze(1)
            span_scores += width_scores

        return span_scores

    def ffnn(self, input_size, num_hidden_layers, hidden_size, output_size, dropout=None):
        ffnn = torch.nn.Sequential()
        for i in range(num_hidden_layers):
            linear_layer = nn.Linear(input_size, hidden_size, bias=True)
            ffnn.add_module("hidden_layer_{}".format(i), linear_layer)
            ffnn.add_module("hidden_layer_{}_relu".format(i), nn.ReLU(inplace=True))
            if dropout is not None:
                ffnn.add_module("hidden_layer_{}_dropout".format(i), nn.Dropout(p=dropout))

            input_size = hidden_size

        output_layer = nn.Linear(input_size, output_size, bias=True)
        ffnn.add_module("output_layer", output_layer)

        for module in ffnn.modules():
            if isinstance(module, torch.nn.Linear):
                nn.init.trunc_normal_(module.weight.data, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        return ffnn
