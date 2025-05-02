# FakeNewsDetection

##  Project Overview
- **Title:** *Exposing the truth with advanced fake news detection powered by NLP*  
- **Goal:** Detect fake news using an LSTM-based NLP system  
- **Importance:** Prevents misinformation, protects public trust, supports informed decisions  

---

## Objectives
- Develop ML model to classify news as real or fake based on text  
- Provide credibility scores and detect misinformation patterns  
- Enable faster, automated detection and improve media literacy  

---

## Data Collection
 - **Dataset Name:** Fake News Detection by Sameer Patel  
- **Platform:** Kaggle  
- **Access:** Public  
- **Fields:**
  - `title`: News headline  
  - `text`: News content  
  - `label`: 1 = fake, 0 = real  


### Dataset Link:
    https://www.kaggle.com/code/therealsampat/fake-news-detection/input
    

## Data Preprocessing

This section details the data cleaning and preparation processes applied to the True and Fake news datasets before training models for fake news detection.

---

### 1. Handling Missing Values

- Converted the `date` column to datetime format using `errors='coerce'`.
- Removed rows with invalid or missing dates.

```python
df['date'] = pd.to_datetime(df['date'], errors='coerce')
ds['date'] = pd.to_datetime(ds['date'], format='mixed', errors='coerce')
df = df.dropna(subset=['date'])
ds = ds.dropna(subset=['date'])
```

---

### 2. Removing Duplicate Records

- Checked and removed duplicate rows from both datasets to prevent model bias.

```python
df = df.drop_duplicates()
ds = ds.drop_duplicates()
```

---

### 3. Data Type Conversion and Consistency

- Ensured the `date` column is in datetime format.
- Verified data types using `.info()`.

```python
print(df.info())
print(ds.info())
```

---

### 4. Text Cleaning

- Removed unwanted characters, HTML tags, and punctuation.
- Converted all text to lowercase.

```python
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9,.!?;:()'\"-]", ' ', text)
    return text.strip().lower()

df['title'] = df['title'].apply(clean_text)
df['text'] = df['text'].apply(clean_text)
ds['title'] = ds['title'].apply(clean_text)
ds['text'] = ds['text'].apply(clean_text)
```

---

### 5. Stopword Removal and Lemmatization

- Used NLTK to remove stopwords.
- Applied lemmatization using WordNet.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['title'] = df['title'].apply(preprocess_text)
df['text'] = df['text'].apply(preprocess_text)
ds['title'] = ds['title'].apply(preprocess_text)
ds['text'] = ds['text'].apply(preprocess_text)
```

---

###  6. Encoding Categorical Variables

- Encoded the `subject` column using label encoding from scikit-learn.

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['subject_encoded'] = encoder.fit_transform(df['subject'])
ds['subject_encoded'] = encoder.transform(ds['subject'])
```

---

