import io
import numpy as np
import six

from sklearn.decomposition import TruncatedSVD


class SimpleEncoder:
    def __init__(
            self,
            word_embeddings: dict,
            word_embedding_dim: int = None,
            preprocessor: callable = lambda s: s,
            tokenizer: callable = lambda s: s.split(),
            word_freq: dict = None,
            weighted: bool = False,
            alpha: float = 1e-3
    ):
        # word embeddings (dict)
        self.word_embeddings = word_embeddings

        # word embedding dim (int)
        self.word_embedding_dim = word_embedding_dim or next(iter(word_embeddings.values())).shape[0]

        # sentence tokenizer (callable)
        self.tokenizer = tokenizer

        # preprocessor (callable)
        self.preprocessor = preprocessor

        # word frequency (callable)
        self.word_freq = word_freq or {}

        # yes/no: tf-idf weighted average
        self.weighted = weighted

        # smoothing alpha
        self.alpha_ = alpha

        # principal components
        self.components_ = None

    def fit(self, sentences, random_state=None):
        """See ```.fit()`` method."""
        self._fit(sentences, random_state=random_state)
        return self

    def fit_transform(self, sentences, random_state=None):
        """Fit the sentence encoder based on few examples.
           - Call `.fit()` to compute the principal components.
           - (Internally) Call `.transform` to transform sentences to 2d array of embeddings.

        Args:
            sentences: either str or a list[str].
            random_state (int): random seed.

        Returns:
            np.array(shape=(len(sentences), self.word_embedding_dim), dtype=np.float32)
        """
        return self._fit(sentences, random_state=random_state)

    def transform(self, sentences) -> np.array:
        """Transform one or more sentences to a 2d embedding matrix.

        Args:
            sentences: either str or a list[str].

        Returns:
            np.array(shape=(len(sentences), self.word_embedding_dim), dtype=np.float32)
        """
        if isinstance(sentences, six.string_types):
            sentences = [sentences]

        emb = np.stack([self._encode(sentence) for sentence in sentences])
        emb = self._remove_pc_projection(emb)
        return emb

    def _fit(self, sentences, random_state=None):
        """

        Args:
            sentences: either str or a list[str].
            random_state (int): random seed.

        Returns:
            np.array(shape=(len(sentences), self.word_embedding_dim), dtype=np.float32)
        """
        self.components_ = None
        emb = self.transform(sentences)

        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=random_state).fit(emb)
        self.components_ = svd.components_

        emb = self._remove_pc_projection(emb)
        return emb

    def _remove_pc_projection(self, embedddings: np.array) -> np.array:
        """Remove principal component projection.

        https://github.com/PrincetonML/SIF/blob/84b5b4c1c1ca20b6af19fc78cae005a1818ec571/src/SIF_embedding.py#L26

        Args:
            embedddings (np.array): embedding 2d array.

        Returns:
            np.array
        """
        if self.components_ is None:
            return embedddings
        return embedddings - embedddings.dot(self.components_.transpose()).dot(self.components_)

    def _encode(self, text: str) -> np.array:
        """Calc a weighted average of word vectors in a sentence to compute its numerical encoding.

        Args:
            text (str): input text.

        Returns:
            np.array(shape=(self.word_embedding_dim,))
        """
        count = 0
        sent_vec = np.zeros(self.word_embedding_dim, dtype=np.float32)
        text = self.preprocessor(text)
        words = self.tokenizer(text)
        for word in words:
            word_vec = self.word_embeddings.get(word)
            if word_vec is None:
                continue
            norm = np.linalg.norm(word_vec)
            if norm > 0:
                word_vec *= (1.0 / norm)
            if self.weighted:
                freq = self.word_freq.get(word, 0.0)
                word_vec *= self.alpha_ / (self.alpha_ + freq)
            sent_vec += word_vec
            count += 1
        if count > 0:
            sent_vec *= (1.0 / count)
        return sent_vec


def build_from_w2v_path(
        w2v_path: str,
        preprocessor: callable = lambda s: s,
        tokenizer: callable = lambda s: s.split(),
        alpha: float = 1e-3
):
    """Construct `SimpleEncoder` from prebuilt word2vec file.

    Args:
        w2v_path (str): path to w2v file (text format).
        preprocessor (callable): preprocssing callable.
        tokenizer (callable): tokenization callable.
        alpha (float): smoothing alpha.

    Returns:
        SentenceEncoder
    """
    fin = io.open(w2v_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    vocab_size, dim = map(int, fin.readline().split())
    word_embeddings = dict()
    for line in fin:
        tokens = line.rstrip().split(' ')
        word_embeddings[tokens[0]] = np.array(tokens[1:], np.float32)

    return SimpleEncoder(
        word_embeddings=word_embeddings,
        word_embedding_dim=dim,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        weighted=False,
        alpha=alpha
    )


def build_from_fasttext_bin(
        model,
        preprocessor: callable = lambda s: s,
        tokenizer: callable = lambda s: s.split(),
        weighted: bool = False,
        alpha: float = 1e-3
):
    """Construct `SimpleEncoder` from `fasttext` unsupervised model in binary format.

    Args:
        model (fasttext.FastText._FastText): word2vec model.
        preprocessor (callable): preprocssing callable.
        tokenizer (callable): tokenization callable.
        weighted (bool: use tf-idf weighted average.
        alpha (float): smoothing alpha.

    Returns:
        SentenceEncoder
    """
    total = 0
    word_count = dict()
    word_embeddings = dict()
    dim = model.get_dimension()

    for word, word_freq in zip(*model.get_words(include_freq=True)):
        total += word_freq
        word_count[word] = word_freq
        word_embeddings[word] = model.get_word_vector(word)
    word_freq = {word: cnt / total for word, cnt in word_count.items()}

    return SimpleEncoder(
        word_embeddings=word_embeddings,
        word_embedding_dim=dim,
        word_freq=word_freq,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        weighted=weighted,
        alpha=alpha
    )
