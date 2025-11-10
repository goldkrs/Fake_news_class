import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.models.distilbert import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import logging
import re

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Function to preprocess text (lowercase, remove special chars, extra spaces)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess the dataset
logger.info('Loading dataset...')
df = pd.read_csv('train.csv')
df = df.fillna('')
df = df.drop_duplicates()

# Preprocess title and text columns
logger.info('Preprocessing text...')
df['title'] = df['title'].apply(preprocess_text)
df['text'] = df['text'].apply(preprocess_text)
df['text'] = df['title'] + ' ' + df['text']
df = df[df['text'].str.strip() != '']

# Split the data into training and validation sets
logger.info('Splitting data into train and validation sets...')
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, stratify=df['label'], random_state=42
)

# Load the tokenizer and model from the saved directory
logger.info('Loading tokenizer and model...')
tokenizer = DistilBertTokenizerFast.from_pretrained('bert_model')
model = DistilBertForSequenceClassification.from_pretrained('bert_model')

# Tokenize the validation texts
logger.info('Tokenizing validation texts...')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

# Define a custom Dataset class for PyTorch
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) 
        return item
    def __len__(self):
        return len(self.labels)

# Create the validation dataset
val_dataset = NewsDataset(val_encodings, val_labels)

# Define the compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='binary'),
        "recall": recall_score(labels, preds, average='binary'),
        "f1": f1_score(labels, preds, average='binary')
    }

# Define dummy training arguments (required by Trainer)
eval_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
    do_train=False,
    do_eval=True
)

# Initialize the Trainer for evaluation
logger.info('Initializing Trainer for evaluation...')
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=val_dataset
)

# Evaluate the model
logger.info('Evaluating model...')
results = trainer.evaluate()
logger.info('Evaluation complete.')

# Print evaluation metrics
print("Evaluation Results:")
for key, value in results.items():
    print(f"{key}: {value}") 