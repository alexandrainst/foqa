---
dataset_info:
- config_name: all-samples
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: original_context
    dtype: string
  - name: question
    dtype: string
  - name: answers
    struct:
    - name: answer_start
      sequence: int64
    - name: text
      sequence: string
  - name: validation
    dtype: string
  - name: context
    dtype: string
  splits:
  - name: train
    num_bytes: 55875621
    num_examples: 10001
  download_size: 6994468
  dataset_size: 55875621
- config_name: default
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: question
    dtype: string
  - name: answers
    struct:
    - name: answer_start
      sequence: int64
    - name: text
      sequence: string
  - name: validation
    dtype: string
  - name: context
    dtype: string
  splits:
  - name: train
    num_bytes: 2187382
    num_examples: 848
  - name: val
    num_bytes: 347924
    num_examples: 128
  - name: test
    num_bytes: 2487219
    num_examples: 1024
  download_size: 2042697
  dataset_size: 5022525
- config_name: incorrect-samples
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: question
    dtype: string
  - name: answers
    struct:
    - name: answer_start
      sequence: int64
    - name: text
      sequence: string
  - name: validation
    dtype: string
  - name: context
    dtype: string
  splits:
  - name: train
    num_bytes: 6241697
    num_examples: 2395
  download_size: 1492991
  dataset_size: 6241697
configs:
- config_name: all-samples
  data_files:
  - split: train
    path: all-samples/train-*
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
- config_name: incorrect-samples
  data_files:
  - split: train
    path: incorrect-samples/train-*
license: cc-by-nc-4.0
task_categories:
- question-answering
language:
- fo
pretty_name: FoQA
size_categories:
- 1K<n<10K
---

# FoQA: Faroese Question Answering Dataset

## Dataset Overview

**FoQA** is a Faroese extractive question answering (also known as reading
comprehension) dataset. It consists of 2,000 question-answer-context triples, with the
contexts coming from Faroese Wikipedia articles. The dataset has been been created in a
two-stage process: First, 10,000 question-answer-context triples were automatically
generated using GPT-4-turbo. These were then manually reviewed by a native Faroese
speaker, resulting in the final 2,000 triples. For more information about the dataset
creation, check out our [paper](missing). All data points are available, even the ones
that were rejected, or not manually validated at all.


## Dataset Versions

We're releasing three versions of the dataset:

- `default`: The default version of the dataset, with 848 training examples, 128
  validation examples, and 1024 test examples. These are the validated examples.
- `all-samples`: All 10,001 examples from the original dataset, including the ones that
  were rejected or not manually validated.
- `incorrect-samples`: The 2,395 examples that were rejected during the manual review
  process.

We acknowledge that the main use of the dataset will be the `default` version, but we
also provide the other versions in case they are useful for research purposes.


## Data Fields

The dataset is formatted in the standard
[SQuAD](https://huggingface.co/datasets/rajpurkar/squad_v2), with the following
features:

- `id`: The URL of the Wikipedia article the context is from.
- `title`: The title of the Wikipedia article the context is from.
- `question`: The question.
- `answers`: The answer to the question, being a dictionary with the following keys:
  - `answer_start`: A list of character offsets in the context where the answer begins.
  - `text`: A list of the corresponding answer texts.
- `validation`: The validation status of the data point.


## License

The dataset is licensed under the non-commercial [CC BY-NC
4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. It is non-commercial as
it contains generated data from OpenAI models.


## Citation

If you use FoQA in your research, please cite our paper:

```bibtex
@article{simonsen2024foqa,
  title={FoQA: A Faroese Question-Answering Dataset},
  author={Simonsen, Annika and Nielsen, Dan Saattrup and Einarsson, Hafsteinn},
  journal={arXiv preprint arXiv:9999.99999},
  year={2024}
}
```
