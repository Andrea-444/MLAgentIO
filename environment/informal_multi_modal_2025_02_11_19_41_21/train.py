from datasets import load_dataset, Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Load dataset
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
train_sentences = train_df['sentence'].tolist()
train_styles = train_df['style'].tolist()
test_sentences = test_df['sentence'].tolist()
test_styles = test_df['style'].tolist()

# Encode labels
label_encoder = LabelEncoder()
train_styles = label_encoder.fit_transform(train_styles)
test_styles = label_encoder.transform(test_styles)

# LSTM Model
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_sequences = sequence.pad_sequences(train_sequences, maxlen=max_len)
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_sequences = sequence.pad_sequences(test_sequences, maxlen=max_len)

lstm_model = Sequential()
lstm_model.add(Embedding(max_words, 128))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(len(np.unique(train_styles)), activation='softmax'))

lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(train_sequences, train_styles, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate LSTM model
lstm_predictions = np.argmax(lstm_model.predict(test_sequences), axis=-1)
lstm_report = classification_report(test_styles, lstm_predictions)
lstm_confusion = confusion_matrix(test_styles, lstm_predictions)

print(lstm_report)
print(lstm_confusion)

# BERT Model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_sentences, truncation=True, padding=True, max_length=128)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_styles,
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_styles,
})

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(train_styles)))
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()
trainer.evaluate()

# BERT Model Evaluation
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
bert_report = classification_report(test_styles, preds)
bert_confusion = confusion_matrix(test_styles, preds)

print(bert_report)
print(bert_confusion)

# Save submission files
submission_file_lstm = open("submission_lstm.txt", "w", encoding="utf-8")
submission_file_lstm.write("\nLSTM Test classification report:\n")
submission_file_lstm.write(lstm_report)
submission_file_lstm.write("\nLSTM Confusion Matrix:\n")
submission_file_lstm.write(str(lstm_confusion))
submission_file_lstm.close()

submission_file_bert = open("submission_bert.txt", "w", encoding="utf-8")
submission_file_bert.write("\nBERT Test classification report:\n")
submission_file_bert.write(bert_report)
submission_file_bert.write("\nBERT Confusion Matrix:\n")
submission_file_bert.write(str(bert_confusion))
submission_file_bert.close()
