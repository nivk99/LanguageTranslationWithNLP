"""
The project is broken down into four steps:
1. Understand the data
2. Preprocess the data
3. Train the model
4. Test the model

fork git: https://github.com/SeanvonB/language-translator
"""
# import
from typing import List
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout
from keras.layers import Embedding
import pickle
from keras.optimizers import Adam, Adamax
from keras.losses import sparse_categorical_crossentropy
from wordcloud import WordCloud
import seaborn as sns
import os
import copy
import re
from string import punctuation




def load_data(path: str) -> List[str]:
    """
    Load dataset
    """
    input_file = os.path.join(path)
    # TODO: Open file and return a stream.  Raise OSError upon failure
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data.split('\n')


def plotting_results(history: Sequential) -> None:
    """
    Plot results of a model trainig
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # TODO: plot - Training and validation accuracy
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    # TODO: plot - Training and validation lossy
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def word_cloud(eng, fra) -> None:
    # TODO: Creating Wordcloud for the English text
    plt.figure(figsize=(15, 12))
    wc = WordCloud(width=600, height=300).generate(' '.join(eng))
    plt.imshow(wc)
    plt.show()

    # TODO: Creating a wordcloud for French text
    plt.figure(figsize=(15, 12))
    wcf = WordCloud(width=600, height=300).generate(' '.join(fra))
    plt.imshow(wcf)
    plt.show()


def dist_plot(x, y) -> None:
    """
    :param x: English sentences
    :param y:French sentences
    :return: void - plot

  Average word count of the english text is about 11-16 words. Maximum reaching 17+.
  While that of the French text seemd to be around same 15-18 words. Maximum reaching around 21+.
    """

    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.distplot([len(wx.split()) for wx in x], ax=axes[0])
    sns.distplot([len(wy.split()) for wy in y], ax=axes[1])
    sns.despine()
    plt.show()


def evaluate_model(model, x, y) -> None:
    """
    evaluate the model
    :param model:
    :param x:Test sentences in English
    :param y:Test sentences in French
    :return: void
    """
    score = model.evaluate(x, y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


def summarize(model) -> None:
    """
    Prints a string summary of the network
    :param model:model
    :return:prints
    """
    model.summary()


def save_model(model) -> None:
    """
    save model and architecture to single file
    :param model:
    :return: void
    """
    model.save("model.h5")
    print("Saved model to disk")


def english_sentences() -> list[str]:
    """
    Loads sentences in English
    :return: list[str]
    """
    # TODO: load
    return load_data('data/en')


def french_sentences() -> list[str]:
    """
    Loads sentences in French
    :return: list[str]
    """
    # TODO: load
    return load_data('data/fr')


def tokenize(x: List[str]):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)

    :examples:
    word_index: {'the': 1, 'quick': 2, 'a': 3, 'brown': 4, 'fox': 5, 'jumps': 6, 'over': 7,
     'lazy': 8, 'dog': 9, 'by': 10, 'jove': 11,'my': 12, 'study': 13, 'of': 14,
      'lexicography': 15, 'won': 16, 'prize': 17, 'this': 18, 'is': 19, 'short': 20, 'sentence': 21}
    Sequence 1 in x
        Input:  The quick brown fox jumps over the lazy dog .
        Output: [1, 2, 4, 5, 6, 7, 1, 8, 9]
    Sequence 2 in x
        Input:  By Jove , my quick study of lexicography won a prize .
        Output: [10, 11, 12, 2, 13, 14, 15, 16, 3, 17]
    Sequence 3 in x
        Input:  This is a short sentence .
        Output: [18, 19, 3, 20, 21]
    """
    x_tk = Tokenizer(char_level=False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences

    :examples:
    length=10
    Sequence 1 in x
        Input:  [1 2 4 5 6 7 1 8 9]
        Output: [1 2 4 5 6 7 1 8 9 0]
    Sequence 2 in x
        Input:  [10 11 12  2 13 14 15 16  3 17]
        Output: [10 11 12  2 13 14 15 16  3 17]
    Sequence 3 in x
        Input:  [18 19  3 20 21]
        Output: [18 19  3 20 21  0  0  0  0  0]
    """
    if length is None:
        length = max([len(sentence) for sentence in x])

    # TODO:Pads sequences to the same length
    return pad_sequences(x, maxlen=length, padding="post")


def preprocess(x: List[str], y: List[str]):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)

    :examples:
    x ['new jersey is sometimes quiet during autumn , and it is snowy in april .']
    y ["new jersey est parfois calme pendant l' automne , et il est neigeux en avril ."]
    preprocess_x [[17, 23, 1, 8, 67, 4, 39, 7, 3, 1, 55, 2, 44]]
    preprocess_y [[35, 34, 1, 8, 67, 37, 11, 24, 6, 3, 1, 112, 2, 50]]
    Pad preprocess_x [[17 23  1  8 67  4 39  7  3  1 55  2 44  0  0]]
    Pad preprocess_y [[ 35 34 1 8 67 37 11 24 6 3 1 112 2 50 0 0 0 0 0 0 0]
    Dimensions preprocess_y [[[ 35] [ 34] [  1] [  8] [ 67] [ 37] [ 11] [ 24] [  6] [  3] [  1] [112] [  2] [ 50] [  0] [  0] [  0] [  0] [  0] [  0] [  0]]

    """
    # TODO:tokenize
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    # TODO: Ambiguity remove
    # for i, v in enumerate(preprocess_x):
    #     arr_x = copy.deepcopy(preprocess_x)
    #     arr_x.remove(v)
    #     if v not in arr_x:
    #         continue
    #     for j in range(i,len(preprocess_x)):
    #         if j == i:
    #             continue
    #         try:
    #             if v == preprocess_x[j] and preprocess_y[i] != preprocess_y[j]:
    #                 preprocess_y[j] = preprocess_y[i]
    #                 preprocess_y.remove(preprocess_y[i])
    #                 preprocess_x.remove(preprocess_x[i])
    #                 arr_x.remove(v)
    #                 if v not in arr_x:
    #                     break
    #         except:
    #             continue

    # TODO:pad
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # TODO:Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size) -> Sequential:
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """

    # TODO: Implement

    # Hyperparameters
    learning_rate = 0.001
    # Build the layers
    model = Sequential()

    # TODO: Adds a layer instance on top of the layer stack
    # Embedding
    model.add(Embedding(english_vocab_size, 256,
                        input_length=output_sequence_length,
                        input_shape=input_shape[1:]))
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))

    # TODO:Configures the model for training.
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adamax(learning_rate),
                  metrics=['accuracy'])
    return model


def final_predictions(x, y, x_tk, y_tk, english_sentences, french_sentences) -> None:
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    :param english_sentences:
    :param french_sentences:
    """

    max_french_sequence_length = y.shape[1]

    # TODO: Add 1 for  token
    english_vocab_size = len(x_tk.word_index) + 1
    french_vocab_size = len(y_tk.word_index) + 1
    x = pad(x, max_french_sequence_length)

    # TODO: Split dataset in train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

    # TODO: Train neural network using model_fina
    model = model_final(
        X_train.shape,
        Y_train.shape[1],
        english_vocab_size,
        french_vocab_size)

    # TODO: store all variables
    pickle.dump({'x': x, 'x_tk': x_tk,
                 'y': y, 'y_ty': y_tk}, open("training_data.pkl", "wb"))

    # TODO: early stopping stops the learning process in the case that the process doesn't progress for 2 epochsor in the case of over fitting to the training data over the validation data
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=2,
                       verbose=1,
                       mode='auto')

    # TODO: An example of the output of fit
    """
    :example:
    Epoch 1/10
    110288/110288 [==============================] - 25s 228us/step - loss: 2.6723 - acc: 0.4934 - val_loss: 1.5712 - val_acc: 0.6215
    Epoch 2/10
    110288/110288 [==============================] - 24s 221us/step - loss: 1.2008 - acc: 0.7022 - val_loss: 0.8650 - val_acc: 0.7777
    """
    # TODO: Trains the model for a fixed number of epochs (iterations on a dataset)
    history_model = model.fit(X_train, Y_train, batch_size=1, epochs=100, callbacks=[es],
                              validation_data=(X_validation, Y_validation))

    # TODO:Creating Wordcloud for the English text and French text
    word_cloud(english_sentences, french_sentences)
    # TODO: plot
    dist_plot(english_sentences, french_sentences)
    # TODO:summarize model
    summarize(model)
    # TODO:plotting model
    plotting_results(history_model)
    # TODO:save model
    save_model(model)
    # TODO: Test the model
    evaluate_model(model, X_test, Y_test)


# TODO: run the file
if __name__ == '__main__':
    # Understand the data
    english = english_sentences()
    french = french_sentences()
    # Preprocess the data
    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
        preprocess(english, french)
    # Initializes the model
    final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer, english
                      , french)
