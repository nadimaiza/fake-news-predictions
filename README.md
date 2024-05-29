
# Fake News Prediction

This project aims to build a machine learning model to classify news articles as real or fake. The script preprocesses the data, trains a Logistic Regression model, and evaluates its performance.

## Prerequisites

Make sure you have the following libraries installed:

- numpy
- pandas
- scikit-learn
- nltk

You can install these using pip:

```bash
pip install numpy pandas scikit-learn nltk
```

## Dataset

The dataset used in this project should be named `fake news_training.csv` and should be placed in the same directory as the script. The dataset is expected to have the following structure:

- `id`: Unique identifier for each news article
- `title`: The title of the news article
- `author`: The author of the news article
- `text`: The full text of the news article
- `label`: The label indicating whether the news is real (0) or fake (1)

## Script Overview

### Importing Libraries

The script begins by importing the necessary libraries:

```python
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
```

### Downloading NLTK Stopwords

The script downloads the NLTK stopwords:

```python
nltk.download("stopwords")
```

### Loading and Preprocessing Data

The data is loaded from the CSV file and some initial preprocessing steps are performed:

```python
data = pd.read_csv("fake news_training.csv")
df = pd.DataFrame(data)

# Checking the shape and head of the dataframe
df.shape
df.head(6)

# Checking the distribution of labels
df.groupby("label")["id"].count()

# Checking for null values
df.isnull().sum()
```

### Text Preprocessing

The script includes functions to preprocess the text data by removing stopwords and performing stemming:

```python
# Function to preprocess text
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    return text

# Applying preprocessing to the dataset
df['text'] = df['text'].apply(preprocess_text)
```

### Splitting the Data

The dataset is split into training and testing sets:

```python
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
```

### Vectorizing the Text Data

The text data is transformed into TF-IDF vectors:

```python
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
```

### Training the Model

A Logistic Regression model is trained on the training data:

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Evaluating the Model

The model's performance is evaluated using the accuracy score:

```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
```

## Running the Script

To run the script, simply execute it in a Python environment:

```bash
python fake_news_prediction.py
```

Make sure the dataset `fake news_training.csv` is in the same directory as the script.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Any contributions are welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
