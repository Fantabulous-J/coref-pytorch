## Pytorch Implementation of c2f-coref Model

### Setup

- pip -r install requirements.txt
- ```./setup_training.sh <ontonotes/path/ontonotes-release-5.0>```.
This assumes that you have access to OntoNotes 5.0. The preprocessed data will be included under ```conll_data```.

### Build Kernels
- ```python setup.py install```. This will build kernel for extracting top spans implemented using the C++ interface
of Pytorch.


### Training
- ```python train.py <experiment>```
- Results are stored in the ```log_root``` directory.
- For getting the result of using SpanBERT-Base and SpanBERT-Large model, use 
```python train.py train_spanbert_base_mention``` and ```python train.py train_spanbert_large_mention```
- Finetuning a SpanBERT large model on OntoNotes requires access to a 32GB GPU, while the base model
can be trained in a 16GB GPU.

### Performance
| Model          | F1 (%) |
|:--------------:|:------:|
| SpanBERT-base  | 77.5   |
| SpanBERT-large | 80.0   |

### Acknowledgement
Many thanks to previous work <https://github.com/mandarjoshi90/coref>.