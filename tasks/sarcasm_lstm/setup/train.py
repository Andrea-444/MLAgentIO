import sys

import pandas as pd

import keras
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing import sequence
from keras.api.models import Sequential
from keras.api.layers import Dense, Embedding, Conv1D, MaxPooling1D, LSTM

from sklearn.metrics import confusion_matrix, classification_report


def get_vocabulary(texts: list[str]) -> list[str]:
    vocabulary = set()
    for text in texts:
        for word in text.split(" "):
            vocabulary.add(word)
    return list(vocabulary)


if __name__ == '__main__':
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    text_len_outliers_limit = 37  # Based on a previous explanatory data analysis

    train_no_outliers = train[train["Text"].str.split().apply(len) < text_len_outliers_limit].copy()
    test_no_outliers = test[test["Text"].str.split().apply(len) < text_len_outliers_limit].copy()

    vocabulary = get_vocabulary(train_no_outliers["Text"].tolist())

    vocabulary_len = len(vocabulary)
    max_vocabulary_size = 6000  # Based on a previous explanatory data analysis

    x_train, y_train = train_no_outliers["Text"], pd.get_dummies(train_no_outliers["Label"])
    x_test, y_test = test_no_outliers["Text"], pd.get_dummies(test_no_outliers["Label"])

    # Text Vectorization

    tokenizer = Tokenizer(num_words=max_vocabulary_size)
    tokenizer.fit_on_texts(x_train.values)

    x_train_embedings = tokenizer.texts_to_sequences(x_train)
    x_test_embedings = tokenizer.texts_to_sequences(x_test)

    num_classes = len(train["Label"].unique())

    max_word_len = 30  # Based on a previous explanatory data analysis

    x_train_embedings_padded = sequence.pad_sequences(x_train_embedings, maxlen=max_word_len)
    x_test_embedings_padded = sequence.pad_sequences(x_test_embedings, maxlen=max_word_len)

    # Model

    model = Sequential([
        Embedding(input_dim=max_vocabulary_size, output_dim=128),
        Conv1D(filters=32, kernel_size=4, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2),
        LSTM(64, dropout=0.1, recurrent_dropout=0.1),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.Recall(), keras.metrics.F1Score(), keras.metrics.Precision()]
    )

    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = log_file

    # Training

    log_file.write("Training:\n")
    history = model.fit(x_train_embedings_padded, y_train, validation_split=0.2, batch_size=16, epochs=4, verbose=2)

    # Predicting

    log_file.write("\nPredicting:\n")
    predicts = model.predict(x_test_embedings_padded, verbose=2)
    test_y_arg = y_test.values.argmax(axis=1)
    predicts_arg = predicts.argmax(axis=1)

    # Classification Report

    log_file.write("\nTest classification report:\n")
    log_file.write(classification_report(test_y_arg, predicts_arg))

    # Classification Report
    log_file.write("\nConfusion Matrix:\n")
    log_file.write(str(confusion_matrix(test_y_arg, predicts_arg)))

    log_file.close()

    # ==== How to save submission =====

    # submission_file = open("submission.txt", "w", encoding="utf-8")
    # submission_file.write("\nTest classification report:\n")
    # submission_file.write(classification_report(test_y_arg, predicts_arg))
    # submission_file.write("\nConfusion Matrix:\n")
    # submission_file.write(str(confusion_matrix(test_y_arg, predicts_arg)))
    # submission_file.close()

    # ==================================
