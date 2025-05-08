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
## Feature Engineering
This section outlines the feature engineering techniques applied to enhance the dataset and extract meaningful patterns for fake news detection.

### 1.Text Cleaning and Normalization

```python
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)) 
    tokens = word_tokenize(text.lower())  
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)
df['clean_text'] = df['text'].apply(preprocess_text)
print(df.head())
```
---

### 2.TF-IDF Vectorization with N-grams

```python
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(df.head())
```
---
 
### 3.Data Feature Extraction

```python
df['date'] = pd.to_datetime(df['date'], dayfirst=False)
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['weekday'] = df['date'].dt.weekday
print(df.head())
```
---

### 4.Sentiment Score 

```python
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity
df['sentiment'] = df['clean_text'].apply(get_sentiment)
print(df.head())
```
---

### 5.Combine All Features

```python
meta_features = df[['day', 'month', 'year', 'weekday', 'sentiment']].reset_index(drop=True)
final_features = pd.concat([meta_features, tfidf_df], axis=1)
print(df.head())
```
---

### 6.Train Word2Vec model 

```python
tokenized = df['clean_text'].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4)
def get_average_vector(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)
df['w2v_vector'] = tokenized.apply(lambda x: get_average_vector(x, w2v_model, 100))
print(df.head())
```
---
```python
import pandas as pd
from sklearn.model\_selection import train\_test\_split
from sklearn.linear\_model import LogisticRegression
from sklearn.metrics import accuracy\_score, classification\_report, confusion\_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

### 1. Load the dataset
```python
df = pd.read\_csv("DataSetForModel.csv")
```

### 2. Convert w2v\_vector string to list of floats
```python
def convert\_vector(vec\_str):
vec\_str = vec\_str.strip("\[]")  # Remove square brackets
return \[float(num) for num in vec\_str.split() if num]  # Split and convert to float


df\['w2v\_vector'] = df\['w2v\_vector'].apply(convert\_vector)
```

### 3. Prepare features (X) and target (y)
```python
X = df\['w2v\_vector'].tolist()
y = df\['label']
```

### 4. Train-test split
```python
X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=42)
```

### 5. Train Logistic Regression model
```python
log\_reg = LogisticRegression(max\_iter=1000)
log\_reg.fit(X\_train, y\_train)
```

### 6. Predict on test data
```python
y\_pred = log\_reg.predict(X\_test)

# Step 7: Evaluation
```python
accuracy = accuracy\_score(y\_test, y\_pred)
conf\_matrix = confusion\_matrix(y\_test, y\_pred)
report = classification\_report(y\_test, y\_pred)
```
### Print results
```python
print(" Accuracy:", accuracy)
print("\n Confusion Matrix:\n", conf\_matrix)
print("\n Classification Report:\n", report)Confusion Matrix:
\[\[4478  209]
\[ 184 4065]]
```

Classification Report:
precision    recall  f1-score   support

```
       0       0.96      0.96      0.96      4687
       1       0.95      0.96      0.95      4249

accuracy                           0.96      8936
```

macro avg       0.96      0.96      0.96      8936
weighted avg       0.96      0.96      0.96      8936
```python
import joblib

joblib.dump(log\_reg, 'LogisticRegression.pkl')
```
### 1. Load Dataset and Preprocess

```python
import pandas as pd

df = pd.read_csv('DataSetForModel.csv')
df['clean_text'] = df['clean_text'].fillna("")  # Ensure no NaNs in text
```
### 2. Define Features and Labels
``` python
X = df['clean_text']
y = df['label']
```
### 3. Split into Train/Test Sets
``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
### 4. TF-IDF Vectorization
``` python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```
### 5. Train the Naive Bayes Model
``` python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
```
### 6. Evaluate the Model
``` python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
Results
Accuracy: 93.13%

Classification Report:

              precision    recall  f1-score   support
           0       0.93      0.94      0.93      4687
           1       0.93      0.92      0.93      4249

    accuracy                           0.93      8936
   macro avg       0.93      0.93      0.93      8936
weighted avg       0.93      0.93      0.93      8936
``` Confusion Matrix:
[[4396  291]
[ 332 3917]]
```
### 1. Load and Preprocess the Dataset

```python
import pandas as pd

df = pd.read_csv('DataSetForModel.csv')

def parse_w2v_vector(vector_str):
    if isinstance(vector_str, str):
        return list(map(float, vector_str.strip('[]').split()))
    else:
        return vector_str

df['w2v_vector'] = df['w2v_vector'].apply(parse_w2v_vector)
```
### 2. Prepare Features and Labels
``` python
X = list(df['w2v_vector'])
y = df['label']
```
### 3. Split into Training and Test Sets
``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 3. Split into Training and Test Sets
``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 4. Train Random Forest Classifier
``` python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
### 5. Make Predictions and Evaluate
``` python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = rf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
### 6. Save Trained Model
``` python
import joblib

joblib.dump(rf, 'RandomForest.pkl')
```
Results
Accuracy: 94%+ (varies based on split and embeddings)

Classification Report: High precision/recall on both classes

Confusion Matrix: Indicates strong generalization capability













