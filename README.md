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

### Merging Cleaned Datasets

```python
import pandas as pd

# Load the cleaned datasets
df_true = pd.read_csv("cleaned_True.csv")
df_fake = pd.read_csv("cleaned_False.csv")

# Add a label column: 1 for real, 0 for fake
df_true['label'] = 1
df_fake['label'] = 0

# Merge the datasets
df = pd.concat([df_true, df_fake], ignore_index=True)

# Save the merged dataset
df.to_csv("Merged_Dataset.csv", index=False)

print(df.head())
print(df.tail())
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

## Exploratory Data Analysis (EDA)

This section involves analyzing the merged dataset using univariate and bivariate methods to gain insights into the distribution, patterns, and relationships in the data.

---

---

### 1. Univariate Analysis

#### a. Label Distribution (Fake vs Real)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load merged dataset
df = pd.read_csv("Merged_Dataset.csv")
sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Distribution of Fake (0) and Real (1) News")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks([0, 1], ['Fake', 'Real'])
plt.show()
```

#### b. Subject Frequency

```python
plt.figure(figsize=(12,6))
sns.countplot(data=df, y='subject', order=df['subject'].value_counts().index)
plt.title("Distribution of Subjects")
plt.xlabel("Count")
plt.ylabel("Subject")
plt.show()
```

#### c. Article Length Distribution

```python
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10,5))
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title("Distribution of Article Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()
```

#### d. Boxplot of Article Length by Label

```python
plt.figure(figsize=(8,5))
sns.boxplot(x='label', y='text_length', data=df)
plt.title("Article Length by Label (Fake vs Real)")
plt.xticks([0, 1], ['Fake', 'Real'])
plt.show()
```

---

### 2. Bivariate Analysis

#### a. Correlation Matrix

```python
correlation = df[['text_length', 'subject_encoded', 'label']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

#### b. Pairplot

```python
sns.pairplot(df[['text_length', 'subject_encoded', 'label']], hue='label', palette='husl')
plt.suptitle("Pairplot: text_length, subject_encoded, label", y=1.02)
plt.show()
```

#### c. Grouped Bar Plot: Mean Article Length by Subject and Label

```python
plt.figure(figsize=(12,6))
sns.barplot(x='subject', y='text_length', hue='label', data=df, ci=None)
plt.title("Average Article Length by Subject and Label")
plt.xticks(rotation=45)
plt.show()
```

---

