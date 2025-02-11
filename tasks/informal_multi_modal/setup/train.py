# All the necessary libraries

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

import keras
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing import sequence
from keras.api.models import Sequential
from keras.api.layers import Dense, Embedding, Conv1D, MaxPooling1D, LSTM

from sklearn.metrics import classification_report, confusion_matrix


if __name__ == '__main__':

    # Generate model/solution from here
    pass

    # ==== Example on how to save submission =====

    # submission_file = open("submission.txt", "w", encoding="utf-8")
    # submission_file.write("\nTest classification report:\n")
    # submission_file.write(classification_report(labels, predicts))
    # submission_file.write("\nConfusion Matrix:\n")
    # submission_file.write(str(confusion_matrix(labels, predicts)))
    # submission_file.close()

    # ==================================
