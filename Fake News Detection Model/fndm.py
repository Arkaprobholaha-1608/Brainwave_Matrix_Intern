import gradio as gr
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
news_df = pd.read_csv('News_Dataset.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df['content']
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
vector = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vector.fit_transform(news_df['content'])

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fit logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

def predict_news(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data).max()
    if prediction[0] == 0:
        return f'The news is likely fake with a confidence of {confidence:.2f}'
    else:
        return f'The news is likely real with a confidence of {confidence:.2f}'

gr.Interface(
    fn=predict_news,
    inputs="text",
    outputs="text",
    title="Fake News Detector"
).launch(share = True)
