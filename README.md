## simple-but-tough-to-beat-examples

A set of examples to demonstrate the performance of "simple but tought to beat" sentence embeddings in different downstream tasks. For simplicity and speed, we use `FastText` to learn word representations from a corpus and use them for the baseline. 


## Quick Setup

Install dependencies

```sh
$ pip install -U pipenv && pipenv install --dev
```

export to $PYTHONPATH

```sh
export PYTHONPATH=$PYTHONPATH:/path/to/simple-but-tough-to-beat-examples
```

## Usage

Load pre-built w2v

```python

from encoder import build_from_w2v_path

sentence_encoder = build_from_w2v_path('wiki-news-300d-1M-subword.vec')
```

Load pre-built `FastText` model

```python

from encoder import build_from_fasttext_bin

sentence_encoder = build_from_fasttext_bin('cc.en.300.bin')
```

Fine-tune

```python

sentence_encoder.fit(corpus)
```

transform sentences to embeddings

```python
corpus = [
	'this is a sentence',
	'this is another sentence',
	...
]

sentence_encoder.fit_transform(corpus)
```

## Examples
  
  - [Fake News Classification](https://github.com/talmago/simple-but-tough-to-beat-examples/blob/master/examples/fake_news.ipynb) - Following this [blog post](https://towardsdatascience.com/using-use-universal-sentence-encoder-to-detect-fake-news-dfc02dc32ae9) we demonstrate roughly the same performance of the "Universal Sentence Encoder" for classification of "fake news".
  - [IMDB Review Sentiment Analysis](https://github.com/talmago/simple-but-tough-to-beat-examples/blob/master/examples/imdb.ipynb) - We show how ~90% accuracy can be achieved with the baseline encoder (SOTA = [XLNet](http://nlpprogress.com/english/sentiment_analysis.html)).


## References

[1] Sanjeev Arora, Yingyu Liang, Tengyu Ma, [*A Simple but Tough-to-Beat Baseline for Sentence Embeddings*](https://openreview.net/forum?id=SyK00v5xx)

```
@article{arora2017asimple, 
	author = {Sanjeev Arora and Yingyu Liang and Tengyu Ma}, 
	title = {A Simple but Tough-to-Beat Baseline for Sentence Embeddings}, 
	booktitle = {International Conference on Learning Representations},
	year = {2017}
}
```

[2] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606)

```
@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```