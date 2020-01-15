import fasttext
import numpy as np
import tempfile

from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential, load_model  # noqa


def train_w2v(sentences, model='skipgram', dim=200, min_count=20, lr=0.015, ws=7, minn=3, maxn=6, epoch=20):
    """train word2vec ( via ``fasttext.unsupervised`` ).

    Args:
        sentences (list-like): list of raw sentences.
        model (str): model name (options are: 'skipgram' and 'cbow').
        dim (int): embedding size. default is 200.
        min_count (int): filter words with less than ``min_count`` occurrences.
        lr (float): learning rate.
        ws (int): window-size.
        minn (int): subword min length (default: 3-char).
        maxn (int): subword max length (default: 6-char).
        epoch (int): num of training epochs.

    Returns:
        ``fasttext.FastText._FastText``
    """
    with tempfile.NamedTemporaryFile(mode='w', prefix='corpus-', suffix='.txt') as f:
        for raw_sentence in sentences:
            f.write(raw_sentence)
            f.write('\n')

        return fasttext.train_unsupervised(input=f.name,
                                           model=model,
                                           dim=dim,
                                           minCount=min_count,
                                           lr=lr,
                                           epoch=epoch,
                                           ws=ws,
                                           minn=minn,
                                           maxn=maxn)


def train_nn(X, y,
             hidden_layers=(128,),
             activation='relu',
             dropout=0.4,
             epochs=20,
             batch_size=32,
             validation_split=None,
             validation_data=None,
             patience=4,
             shuffle=True,
             optimizer='adam',
             pt='model.h5'):
    """train a classification neural-net.

    Args:
        X (array-like): 1d or 2d array of features.
        y (array-like): 1d array of labels.
        hidden_layers (tuple): hidden layer size.
        activation (str): activation, default is `relu`.
        dropout (float): dropout rate. default is 0.4.
        epochs (int): training steps. default is 20.
        batch_size (int): batch size. default is 32.
        validation_split (float): validation split. default is 0.1.
        validation_data (tuple): tuple of (X_valid, y_valid).
        patience (int): num of "bad epochs" to wait before stopping the training.
        shuffle (bool): shuffle training data before each epoch. default is true.
        optimizer (str): optimizer name. default is Adam.
        pt (str): checkpoint file path.

    Returns:
        ``keras.models.Sequential``
    """
    n_classes = np.unique(y).shape[0]
    model = Sequential()

    input_dim = X.shape[1]
    for hidden_layer in hidden_layers:
        model.add(Dense(hidden_layer, activation=activation, input_dim=input_dim))
        if dropout:
            model.add(Dropout(dropout))
        input_dim = hidden_layer

    model.add(Dense(1 if n_classes == 2 else n_classes, activation='sigmoid' if n_classes == 2 else 'softmax'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy' if n_classes == 2 else 'categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(),
        EarlyStopping(patience=patience),
        ModelCheckpoint(filepath=pt, save_best_only=True)
    ]

    model.fit(X,
              y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_split,
              validation_data=validation_data,
              shuffle=shuffle,
              callbacks=callbacks)

    return model
