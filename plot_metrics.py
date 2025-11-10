import matplotlib.pyplot as plt
import torch
from transformers.models.distilbert import DistilBertForSequenceClassification, DistilBertTokenizerFast
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
import os
import json
import sys

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/plotting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper function to load logs from a JSON file if available
def load_trainer_logs(log_path='logs/trainer_log.json'):
    if os.path.exists(log_path):
        logger.info(f'Loading logs from {log_path}')
        with open(log_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f'Log file {log_path} not found. Plots will be empty or use placeholder data.')
        return []

# Load logs (update log_path if you save logs elsewhere)
logs = load_trainer_logs()

# Extract metrics from logs if available
train_loss = [entry['loss'] for entry in logs if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in logs if 'eval_loss' in entry]
eval_accuracy = [entry['eval_accuracy'] for entry in logs if 'eval_accuracy' in entry]
eval_f1 = [entry['eval_f1'] for entry in logs if 'eval_f1' in entry]
eval_steps = [i for i, entry in enumerate(logs) if 'eval_accuracy' in entry]
precision = [entry['eval_precision'] for entry in logs if 'eval_precision' in entry]
recall = [entry['eval_recall'] for entry in logs if 'eval_recall' in entry]

# Plot evaluation metrics over time
plt.figure(figsize=(10,6))
plt.plot(eval_steps, eval_accuracy, label='Accuracy')
plt.plot(eval_steps, precision, label='Precision')
plt.plot(eval_steps, recall, label='Recall')
plt.plot(eval_steps, eval_f1, label='F1 Score')
plt.xlabel("Eval Steps")
plt.ylabel("Metric")
plt.title("Evaluation Metrics Over Time")
plt.legend()
plt.grid(True)
plt.savefig("evaluation_metrics.png")
plt.show()

# Plot loss curves
plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(eval_loss, label='Eval Loss')
plt.xlabel('Logging Steps')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

# For confusion matrix, reload the validation set and model, then predict and plot
logger.info('Preparing confusion matrix...')
df = pd.read_csv('train.csv')
df = df.fillna('')
df['text'] = df['title'] + ' ' + df['text']
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, stratify=df['label'], random_state=42
)
tokenizer = DistilBertTokenizerFast.from_pretrained('bert_model')
model = DistilBertForSequenceClassification.from_pretrained('bert_model')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) 
        return item
    def __len__(self):
        return len(self.labels)
val_dataset = NewsDataset(val_encodings, val_labels)
trainer = Trainer(model=model)
preds = trainer.predict(val_dataset)
y_true = preds.label_ids
y_pred = preds.predictions.argmax(-1)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print(sys.executable) 