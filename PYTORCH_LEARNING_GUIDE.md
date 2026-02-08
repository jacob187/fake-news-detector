# Fake News Detector: App Overview & PyTorch Learning Guide

## Table of Contents

1. [How This App Works](#how-this-app-works)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Pros and Cons](#pros-and-cons)
4. [Learning PyTorch From This App](#learning-pytorch-from-this-app)
5. [Core ML Concepts Illustrated](#core-ml-concepts-illustrated)
6. [Exercises: Build Your Own PyTorch Classifiers](#exercises-build-your-own-pytorch-classifiers)

---

## How This App Works

This is a **binary text classification** system that predicts whether a news article is **real** or **fake**. It has three layers:

### 1. Data Pipeline

```
Raw CSVs (true.csv + fake.csv)
    → Label (1=real, 0=fake)
    → Merge & shuffle
    → Clean text (remove URLs, HTML, citations, newlines)
    → Remove duplicates & nulls
    → Save as pickle
```

- **Source**: Kaggle dataset — 21,417 real + 23,481 fake articles (44,689 total)
- **Preprocessing** (`utils/clean_text.py`): regex-based removal of noise (URLs, HTML tags, bracket citations)
- **Splitting** (`models/train_model.py`): 70/30 train-temp/test, then 67/33 train/validation from the temp set — roughly **47% train, 23% val, 30% test**

### 2. Model (Transfer Learning with DistilBERT)

```
Input text
    → DistilBertPreprocessor (tokenize, pad to 512 tokens)
    → DistilBERT backbone (FROZEN — pretrained weights)
    → Classification head (2 output logits)
    → SparseCategoricalCrossentropy loss
    → Adam optimizer (lr=5e-4)
```

- Built with **TensorFlow/Keras 3.0** and **KerasNLP**
- The DistilBERT backbone (`distil_bert_base_en_uncased`) is **frozen** — only the classification head trains
- Trained for **1 epoch** with **batch size 128**
- Achieves **95.52% validation accuracy** and **F1 score of 97**

### 3. Web Interface (Streamlit)

```
User pastes article text
    → clean_text()
    → model.predict()
    → argmax → "fake" or "real"
    → softmax → confidence score
    → Display result + word cloud visualization
```

- Streamlit app with two pages: **Home** (classifier) and **White Paper** (technical docs)
- Model is loaded once with `@st.cache_resource` for performance
- Generates a word cloud of the submitted article for visual inspection

### End-to-End Data Flow

```
Article text → Regex cleaning → DistilBERT tokenizer → Frozen DistilBERT encoder
→ Classification head → Logits [fake_score, real_score] → argmax → Label
                                                        → softmax → Confidence %
```

---

## Architecture Deep Dive

### Key Files and What They Do

| File | Purpose |
|------|---------|
| `models/define_model.py` | Defines the DistilBERT classifier architecture |
| `models/train_model.py` | Data splitting + training loop (`model.fit()`) |
| `models/fake_news_distilbert_model_main.py` | Orchestrates the full training pipeline |
| `models/test_model.py` | Evaluation: predictions, confusion matrix, F1 |
| `utils/clean_text.py` | Regex-based text preprocessing |
| `utils/probability_calculations.py` | Softmax, sigmoid, confidence scoring |
| `utils/create_training_dataset.py` | Loads pickled data + shuffles |
| `streamlit/Home.py` | Web UI: input, prediction, visualization |
| `notebooks/*.ipynb` | EDA + data preprocessing |

### The Model Definition (Simplified)

```python
# models/define_model.py
preprocessor = DistilBertPreprocessor.from_preset("distil_bert_base_en_uncased",
                                                   sequence_length=512)
classifier = DistilBertClassifier.from_preset("distil_bert_base_en_uncased",
                                               preprocessor=preprocessor,
                                               num_classes=2,
                                               activation=None)  # Raw logits
classifier.backbone.trainable = False  # FREEZE the pretrained layers

classifier.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=5e-4),
    metrics=[SparseCategoricalAccuracy()]
)
```

### The Training Loop (Simplified)

```python
# models/train_model.py
classifier.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=1,
    batch_size=128
)
```

---

## Pros and Cons

### Pros

| Strength | Detail |
|----------|--------|
| **High accuracy with minimal training** | 95.5% accuracy in 1 epoch — transfer learning is powerful |
| **Clean project structure** | Separated concerns: model definition, training, utils, web app |
| **Practical end-to-end pipeline** | Goes from raw CSV → trained model → interactive web app |
| **Frozen backbone = fast training** | Only the classification head trains, so it works on consumer hardware |
| **Good preprocessing** | Text cleaning handles common web noise (URLs, HTML, citations) |
| **Documented** | Includes a whitepaper, intro content, and README |
| **Reproducible** | Fixed `random_state=42`, pinned dependencies, `.python-version` |
| **Lightweight UI** | Streamlit is appropriate for ML demos — minimal frontend code |

### Cons

| Weakness | Detail |
|----------|--------|
| **Only 1 epoch** | The model likely undertrained — more epochs with early stopping could improve results |
| **Validation data leak** | `fit_data()` passes `(X_test, y_test)` as `validation_data` but `X_val` is computed and never used — test set is used for both validation and evaluation |
| **No data augmentation** | Text augmentation (synonym replacement, back-translation) could improve robustness |
| **No learning rate scheduling** | A scheduler (e.g., cosine annealing, warmup) would help convergence |
| **No early stopping or checkpointing** | Training has no safeguard against overfitting or crash recovery |
| **Limited evaluation** | F1 is computed on only 1,000 test samples, not the full test set |
| **Dataset bias** | True/fake articles come from different sources with different subject distributions — the model may learn source style rather than truthfulness |
| **No cross-validation** | A single train/test split doesn't measure variance in performance |
| **Hardcoded paths** | File paths like `"../data/processed/data.pk1"` are brittle — no config system |
| **Binary classification only** | Real-world fake news exists on a spectrum (misleading, satire, propaganda, etc.) |
| **No input validation in the web app** | Empty or extremely short inputs aren't handled gracefully |
| **Large model for the task** | DistilBERT has 66M parameters; simpler models could achieve competitive results on this dataset |

---

## Learning PyTorch From This App

This app uses TensorFlow/Keras. Below is a concept-by-concept translation showing how to build the same system in PyTorch. This is the core of your learning path.

### Concept 1: Transfer Learning (Frozen Backbone + Classification Head)

**What this app does (Keras):**
```python
classifier = DistilBertClassifier.from_preset("distil_bert_base_en_uncased",
                                               num_classes=2)
classifier.backbone.trainable = False
```

**PyTorch equivalent:**
```python
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Freeze the backbone
for param in model.distilbert.parameters():
    param.requires_grad = False
```

**What you learn**: In PyTorch, `requires_grad = False` is how you freeze parameters. Hugging Face `transformers` is the PyTorch ecosystem equivalent of KerasNLP.

---

### Concept 2: Tokenization

**What this app does**: KerasNLP `DistilBertPreprocessor` handles tokenization automatically.

**PyTorch equivalent:**
```python
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize a batch
encoded = tokenizer(
    texts,                  # List of strings
    padding=True,           # Pad to longest in batch
    truncation=True,        # Truncate to max_length
    max_length=512,
    return_tensors="pt"     # Return PyTorch tensors
)
# encoded["input_ids"]      → token IDs
# encoded["attention_mask"] → 1 for real tokens, 0 for padding
```

**What you learn**: PyTorch tokenizers are explicit — you see the input_ids and attention_mask tensors directly. This is more transparent than Keras preprocessing.

---

### Concept 3: DataLoader (Batching)

**What this app does**: Keras `fit()` handles batching internally with `batch_size=128`.

**PyTorch equivalent:**
```python
from torch.utils.data import Dataset, DataLoader

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

**What you learn**: PyTorch makes you build the `Dataset` and `DataLoader` yourself. This gives you full control over how data is batched, shuffled, and fed to the model.

---

### Concept 4: The Training Loop

**What this app does (Keras — one line):**
```python
classifier.fit(x=X_train, y=y_train, epochs=1, batch_size=128)
```

**PyTorch equivalent (explicit loop):**
```python
import torch
import torch.nn as nn
from torch.optim import Adam

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
optimizer = Adam(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        # 1. Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits

        # 2. Compute loss
        loss = loss_fn(logits, batch["label"])

        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 4. Update weights
        optimizer.step()
```

**What you learn**: This is the most important PyTorch concept. The explicit loop shows you exactly what happens:
1. **Forward pass** — data flows through the model
2. **Loss computation** — measures how wrong predictions are
3. **Backward pass** — `loss.backward()` computes gradients via autograd
4. **Parameter update** — `optimizer.step()` adjusts weights using gradients

---

### Concept 5: Loss Functions

**What this app uses**: `SparseCategoricalCrossentropy(from_logits=True)` — integer labels, raw logits.

**PyTorch equivalent:**
```python
loss_fn = nn.CrossEntropyLoss()  # Expects raw logits + integer labels
# This combines LogSoftmax + NLLLoss internally
```

**Other loss functions to know:**
```python
nn.BCEWithLogitsLoss()   # Binary classification with 1 output logit
nn.MSELoss()             # Regression
nn.NLLLoss()             # When you apply log_softmax yourself
```

---

### Concept 6: Softmax and Predictions

**What this app does** (`utils/probability_calculations.py`):
```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
```

**PyTorch equivalent:**
```python
import torch.nn.functional as F

# From logits to probabilities
probs = F.softmax(logits, dim=-1)

# From logits to predicted class
predicted_class = torch.argmax(logits, dim=-1)

# Confidence = max probability
confidence = probs.max(dim=-1).values
```

**What you learn**: `F.softmax` is numerically stable and operates on tensors. Understanding softmax is essential — it converts raw scores into a probability distribution.

---

### Concept 7: Evaluation

**What this app does** (`models/test_model.py`): Confusion matrix + F1 score via sklearn.

**PyTorch evaluation loop:**
```python
from sklearn.metrics import classification_report, confusion_matrix

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():  # Disable gradient computation for efficiency
    for batch in test_loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))
print(confusion_matrix(all_labels, all_preds))
```

**What you learn**: `model.eval()` disables dropout/batchnorm training behavior. `torch.no_grad()` saves memory during inference. sklearn metrics work the same regardless of framework.

---

### Concept 8: Saving and Loading Models

**What this app does**: `classifier.save("./builds/model.keras")`

**PyTorch equivalent:**
```python
# Save
torch.save(model.state_dict(), "model.pt")

# Load
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

**What you learn**: PyTorch saves the `state_dict` (just the weights), not the full model graph. You must reconstruct the architecture first, then load weights into it.

---

## Core ML Concepts Illustrated

This app demonstrates several foundational ML concepts. Here's what each one means and how to think about it:

### 1. Transfer Learning
**What it is**: Using a model pretrained on a massive dataset (DistilBERT was trained on Wikipedia + BookCorpus) and adapting it to a specific task.

**Why it works**: Language understanding transfers across tasks. DistilBERT already knows grammar, semantics, and context — you just teach it what "fake" vs "real" looks like.

**Analogy**: Like hiring a fluent English speaker and training them to be a fact-checker, instead of teaching someone English from scratch.

### 2. Binary Classification
**What it is**: Predicting one of two classes (fake=0, real=1).

**The math**: Model outputs 2 logits → softmax → probabilities → argmax → class.

**Generalization**: Multi-class classification is the same pattern with more output classes.

### 3. Gradient Descent and Backpropagation
**What happens during training**:
1. Forward pass: compute predictions
2. Loss: measure error between predictions and truth
3. Backward pass: compute how each weight contributed to the error (gradients)
4. Update: adjust weights in the direction that reduces error

**This app**: Adam optimizer handles step 4 with adaptive learning rates per parameter.

### 4. Overfitting vs. Underfitting
- **Underfitting**: Model is too simple or trained too little (this app trains only 1 epoch — potentially underfitting)
- **Overfitting**: Model memorizes training data instead of learning patterns
- **Mitigations**: validation set monitoring, early stopping, dropout, regularization

### 5. Train/Validation/Test Split
- **Train**: Model learns from this data
- **Validation**: Used during training to monitor performance (tune hyperparameters)
- **Test**: Used once at the end for final evaluation
- **Bug in this app**: The test set is used as validation data during training, which is a data leak

### 6. Tokenization
**What it does**: Converts raw text into numerical IDs that models can process.
- "The news is fake" → [101, 1996, 2739, 2003, 8275, 102]
- Includes special tokens: [CLS] at start, [SEP] at end
- Handles padding (all sequences same length) and truncation (cap at max_length)

### 7. Confusion Matrix
```
                Predicted Fake    Predicted Real
Actual Fake         TP                FN
Actual Real         FP                TN
```
- **Precision**: Of all "fake" predictions, how many were actually fake?
- **Recall**: Of all actual fake articles, how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall

---

## Exercises: Build Your Own PyTorch Classifiers

Use these exercises to progressively build your PyTorch skills, using this app's concepts as a foundation.

### Exercise 1: Sentiment Classification from Scratch (Beginner)

Build a simple text classifier without pretrained transformers.

```python
import torch
import torch.nn as nn

class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)       # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)           # Average pooling over sequence
        x = self.relu(self.fc1(x))
        x = self.fc2(x)             # Raw logits
        return x
```

**Task**: Train this on the IMDB dataset (`torchtext` or Hugging Face `datasets`). Compare accuracy to a random baseline.

**Concepts practiced**: Embedding layers, forward pass, DataLoader, training loop.

---

### Exercise 2: CNN Text Classifier (Intermediate)

Add convolutional layers for better feature extraction.

```python
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3)
        self.conv4 = nn.Conv1d(embed_dim, num_filters, kernel_size=4)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5)
        self.fc = nn.Linear(num_filters * 3, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        c3 = F.relu(self.conv3(x)).max(dim=2).values
        c4 = F.relu(self.conv4(x)).max(dim=2).values
        c5 = F.relu(self.conv5(x)).max(dim=2).values
        x = torch.cat([c3, c4, c5], dim=1)
        x = self.dropout(x)
        return self.fc(x)
```

**Task**: Train on the same fake news dataset this app uses. Compare to a bag-of-words baseline. Add early stopping.

**Concepts practiced**: Conv1d, multi-scale feature extraction, max pooling, dropout, early stopping.

---

### Exercise 3: LSTM Classifier (Intermediate)

Use recurrent layers for sequential text understanding.

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Concat both directions
        return self.fc(hidden)
```

**Task**: Train on fake news data. Experiment with `num_layers`, `dropout`, and bidirectional settings. Plot training vs validation loss curves.

**Concepts practiced**: LSTM, hidden states, bidirectional RNNs, learning curves.

---

### Exercise 4: Fine-tune DistilBERT in PyTorch (Advanced)

Reproduce this app's exact approach in PyTorch:

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Freeze backbone (like this app does)
for param in model.distilbert.parameters():
    param.requires_grad = False

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-4
)
```

**Task**: Train on the same Kaggle fake news dataset. Then **unfreeze** the backbone and fine-tune the full model with a lower learning rate (2e-5). Compare frozen vs. unfrozen accuracy.

**Concepts practiced**: Transfer learning, parameter freezing, differential learning rates, Hugging Face `transformers`.

---

### Exercise 5: Multi-class News Classifier (Advanced)

Extend binary classification to multi-class:

```python
# Instead of fake/real, classify news by subject:
# Politics, World, Left-leaning, Government, US, Middle East
# Use the 'subject' column from the original dataset

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=6
)
loss_fn = nn.CrossEntropyLoss()  # Works for multi-class automatically
```

**Task**: Handle class imbalance with weighted loss or oversampling. Report per-class precision/recall. Visualize the confusion matrix.

**Concepts practiced**: Multi-class classification, class imbalance handling, weighted loss functions.

---

### Exercise 6: Build an Image Classifier (Branching Out)

Apply the same classification patterns to images:

```python
import torchvision.models as models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        # Freeze backbone (same pattern as this app!)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Replace classification head
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
```

**Task**: Classify CIFAR-10 images. Notice how the pattern is identical to NLP: frozen pretrained backbone + trainable classification head.

**Concepts practiced**: Vision models, torchvision, data augmentation (transforms), the universality of transfer learning.

---

### Recommended Learning Path

```
Week 1-2: Exercise 1 (Simple classifier + training loop fundamentals)
Week 3-4: Exercise 2 (CNNs) or Exercise 3 (LSTMs)
Week 5-6: Exercise 4 (Reproduce this app in PyTorch)
Week 7-8: Exercise 5 (Multi-class) + Exercise 6 (Vision)
```

### Essential PyTorch Resources

- **PyTorch official tutorials**: https://pytorch.org/tutorials/
- **Hugging Face NLP course**: https://huggingface.co/learn/nlp-course
- **fast.ai**: https://course.fast.ai/ (practical deep learning)
- **Andrej Karpathy's "Neural Networks: Zero to Hero"**: YouTube series

### Key PyTorch Differences from TensorFlow/Keras

| Aspect | Keras (this app) | PyTorch |
|--------|------------------|---------|
| Training | `model.fit()` one-liner | Explicit loop (forward, loss, backward, step) |
| Freezing | `layer.trainable = False` | `param.requires_grad = False` |
| Inference mode | Automatic | `model.eval()` + `torch.no_grad()` |
| Data pipeline | Built into `fit()` | `Dataset` + `DataLoader` classes |
| Model definition | Functional/Sequential API | Subclass `nn.Module`, define `forward()` |
| Saving | `model.save()` (full model) | `torch.save(model.state_dict())` (weights only) |
| Gradient clearing | Automatic | Manual `optimizer.zero_grad()` |
| GPU usage | Automatic | Explicit `.to(device)` |

The explicit nature of PyTorch is harder at first but gives you deeper understanding of what's actually happening during training.
