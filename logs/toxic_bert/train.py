from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    
    train_dataset = load_dataset('csv', data_files='data/train.csv')['train']
    test_dataset = load_dataset('csv', data_files='data/test.csv')['train']
    
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_dataset['Label'])
    test_labels = label_encoder.transform(test_dataset['Label'])
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_encodings = tokenizer(train_dataset['Text'], truncation=True, padding=True)
    test_encodings = tokenizer(test_dataset['Text'], truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    trainer.train()

    preds = trainer.predict(test_dataset)
    predictions = preds.predictions.argmax(axis=-1)
    
    report = classification_report(test_labels, predictions)
    matrix = confusion_matrix(test_labels, predictions)

    submission_file = open("submission.txt", "w", encoding="utf-8")
    submission_file.write("\nTest classification report:\n")
    submission_file.write(report)
    submission_file.write("\nConfusion Matrix:\n")
    submission_file.write(str(matrix))
    submission_file.close()