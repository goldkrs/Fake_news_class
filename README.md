Here’s a clean and professional README.md file (without license section):


---

#  Fake News Detection using DistilBERT

A transformer-based fake news classification system built with **DistilBERT**, **PyTorch**, and **Hugging Face Transformers**.  
This project enables model evaluation and real-time prediction to detect fake vs real news articles.

---

## Features
- Fine-tuned **DistilBERT** model for binary classification (Real / Fake)
- Evaluation with metrics: Accuracy, Precision, Recall, F1-score
- Interactive user input for real-time prediction
- Text preprocessing (lowercasing, punctuation removal)
- Logging and reproducible results

---

##  Project Structure

├── check_news_bert.py      # Script for real-time news prediction ├── evaluate_bert.py        # Script for model evaluation ├── bert_model/             # Directory containing fine-tuned DistilBERT model ├── train.csv               # Training dataset └── logs/                   # Evaluation logs

---

 Requirements
Install the necessary dependencies:
```bash
pip install torch transformers scikit-learn pandas


---

 Usage

Evaluate Model Performance

Make sure your fine-tuned model is stored in the bert_model/ folder and your dataset is named train.csv.

Run:

python evaluate_bert.py

This evaluates the model and displays metrics like accuracy, precision, recall, and F1-score.


---

 Check News Manually

Run the following command for real-time fake news detection:

python check_news_bert.py

You will be prompted to enter a news title and body, and the model will predict:

REAL NEWS  
or
FAKE NEWS  


---

 Evaluation Metrics

Metric	Description

Accuracy	Overall correct predictions
Precision	Correctly predicted fake news out of all predicted fakes
Recall	Correctly detected fake news out of all actual fakes
F1-score	Harmonic mean of precision and recall



---

 Model Details

Model: DistilBertForSequenceClassification

Tokenizer: DistilBertTokenizerFast

Max sequence length: 256–512 tokens

Framework: PyTorch and Hugging Face Transformers



---

 Author

HemKishor
 [goldkrishna2005@gmail.com]
https://github.com/goldkrs


---

 Acknowledgments

Hugging Face Transformers

PyTorch

Scikit-learn

