import json
import random

import util
from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset

from transformers import BertTokenizer


class CoNLLCorefResolution(object):
    def __init__(self, doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                 cluster_ids, sentence_map, subtoken_map):
        self.doc_key = doc_key
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.text_len = text_len
        self.speaker_ids = speaker_ids
        self.genre = genre
        self.gold_starts = gold_starts
        self.gold_ends = gold_ends
        self.cluster_ids = cluster_ids
        self.sentence_map = sentence_map
        self.subtoken_map = subtoken_map


class CoNLLDataset(Dataset):
    def __init__(self, features: List[CoNLLCorefResolution], config, sign="train"):
        self.features = features
        self.config = config
        self.sign = sign

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature: CoNLLCorefResolution = self.features[item]
        example = (feature.doc_key, feature.input_ids, feature.input_mask, feature.text_len, feature.speaker_ids,
                   feature.genre, feature.gold_starts, feature.gold_ends, feature.cluster_ids, feature.sentence_map,
                   feature.subtoken_map)
        if self.sign == 'train' and len(example[1]) > self.config["max_training_sentences"]:
            example = truncate_example(*example, self.config)

        return example


class CoNLLDataLoader(object):
    def __init__(self, config, tokenizer, mode="train"):
        if mode == "train":
            self.train_batch_size = 1
            self.eval_batch_size = 1
            self.test_batch_size = 1
        else:
            self.test_batch_size = 1

        self.config = config
        self.tokenizer = tokenizer
        self.genres = {g: i for i, g in enumerate(config["genres"])}

    def convert_examples_to_features(self, data_path):
        with open(data_path) as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]

        data_instances = []
        for example in examples:
            data_instances.append(tensorize_example(example, self.config, self.tokenizer, self.genres))

        return data_instances

    def get_dataloader(self, data_sign="train"):
        if data_sign == 'train':
            features = self.convert_examples_to_features(self.config['train_path'])
            dataset = CoNLLDataset(features, self.config, sign='train')
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size, num_workers=16,
                                    collate_fn=collate_fn)
        elif data_sign == 'eval':
            features = self.convert_examples_to_features(self.config['eval_path'])
            dataset = CoNLLDataset(features, self.config, sign='eval')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.eval_batch_size, num_workers=16,
                                    collate_fn=collate_fn)
        else:
            features = self.convert_examples_to_features(self.config['test_path'])
            dataset = CoNLLDataset(features, self.config, sign='test')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size, num_workers=16,
                                    collate_fn=collate_fn)

        return dataloader


def tensorize_example(example: dict, config: dict, tokenizer: BertTokenizer, genres: dict) -> CoNLLCorefResolution:
    clusters = example["clusters"]
    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
    cluster_ids = [0] * len(gold_mentions)
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
    cluster_ids = torch.tensor(cluster_ids, dtype=torch.int64)

    sentences = example["sentences"]
    num_words = sum(len(s) + 2 for s in sentences)
    speakers = example["speakers"]
    speaker_dict = util.get_speaker_dict(util.flatten(speakers), config['max_num_speakers'])

    max_sentence_length = config['max_segment_len']
    text_len = torch.tensor([len(s) for s in sentences], dtype=torch.int64)

    input_ids, input_mask, speaker_ids = [], [], []
    for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
        sentence = ['[CLS]'] + sentence + ['[SEP]']
        sent_input_ids = tokenizer.convert_tokens_to_ids(sentence)
        sent_input_mask = [-1] + [1] * (len(sent_input_ids) - 2) + [-1]
        sent_speaker_ids = [1] + [speaker_dict.get(s, 3) for s in speaker] + [1]
        while len(sent_input_ids) < max_sentence_length:
            sent_input_ids.append(0)
            sent_input_mask.append(0)
            sent_speaker_ids.append(0)
        input_ids.append(sent_input_ids)
        speaker_ids.append(sent_speaker_ids)
        input_mask.append(sent_input_mask)
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    input_mask = torch.tensor(input_mask, dtype=torch.int64)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.int64)
    assert num_words == torch.sum(torch.abs(input_mask)), (num_words, torch.sum(torch.abs(input_mask)))

    doc_key = example["doc_key"]
    subtoken_map = torch.tensor(example.get("subtoken_map", None), dtype=torch.int64)
    sentence_map = torch.tensor(example['sentence_map'], dtype=torch.int64)
    genre = genres.get(doc_key[:2], 0)
    genre = torch.tensor([genre], dtype=torch.int64)
    gold_starts, gold_ends = tensorize_mentions(gold_mentions)

    return CoNLLCorefResolution(doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                                cluster_ids, sentence_map, subtoken_map)


def tensorize_mentions(mentions):
    if len(mentions) > 0:
        starts, ends = zip(*mentions)
    else:
        starts, ends = [], []

    starts = torch.tensor(starts, dtype=torch.int64)
    ends = torch.tensor(ends, dtype=torch.int64)
    return starts, ends


def truncate_example(doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                     cluster_ids, sentence_map, subtoken_map, config):
    max_training_sentences = config["max_training_sentences"]
    num_sentences = input_ids.size()[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
    speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    sentence_map = sentence_map[word_offset: word_offset + num_words]
    subtoken_map = subtoken_map[word_offset: word_offset + num_words]
    gold_spans = (gold_ends >= word_offset) & (gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]

    return (doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids,
            sentence_map, subtoken_map)


def collate_fn(data):
    data = zip(*data)
    data = [x[0] for x in data]
    return data