
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
 

# %%
#! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz

# %% [markdown]
# ## Read Data

# %%
tsv_file = 'amazon_reviews_us_Office_Products_v1_00.tsv.gz'
df_full = pd.read_csv(tsv_file,compression='gzip',sep='\t',on_bad_lines='warn')


# %% [markdown]
# ## Keep Reviews and Ratings

# %%
df = df_full[['review_body','star_rating']].copy()
df.rename(columns={'review_body': 'Review', 'star_rating': 'Rating'}, inplace=True)
print(df.head())
print(df.shape)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# %% [markdown]
#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# %%
count_negative = (df['Rating'] <= 2).sum()
count_neutral  = (df['Rating'] == 3).sum()
count_positive = (df['Rating'] > 3).sum()
print("Negative reviews:", count_negative)
print("Neutral reviews:", count_neutral)
print("Positive reviews:", count_positive)

df = df[df['Rating'] != 3]
print("\nData shape after discarding rating=3:", df.shape)
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x > 3 else 0)

df_neg = df[df['Sentiment'] == 0]
df_pos = df[df['Sentiment'] == 1]

df_neg_sample = df_neg.sample(n=20000, random_state=42)
df_pos_sample = df_pos.sample(n=20000, random_state=42)
df_downsized = pd.concat([df_neg_sample, df_pos_sample], ignore_index=True)



# %% [markdown]
# # Data Cleaning
# 
# 

# %%
df_downsized.dropna(subset=['Review'], inplace=True)     
df_downsized['Review'] = df_downsized['Review'].astype(str)

avg_length_before_cleaning = df_downsized['Review'].apply(len).mean()
print("Average length (in characters) before cleaning:", avg_length_before_cleaning)

df_downsized['Review'] = df_downsized['Review'].str.lower()

def remove_html_and_urls(text):
    # Remove HTML tags using BeautifulSoup
    text_no_html = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    
    # Remove URLs using regex
    # This pattern matches http://, https://, or www. links
    text_no_url = re.sub(r'(https?://\S+|www\.\S+)', '', text_no_html)
    
    return text_no_url

df_downsized['Review'] = df_downsized['Review'].apply(remove_html_and_urls)

df_downsized['Review'] = df_downsized['Review'].str.replace('[^a-z]', ' ', regex=True)

df_downsized['Review'] = df_downsized['Review'].str.split().str.join(' ')

contractions_dict = {
    "won't": "will not",
    "can't": "cannot",
    "don't": "do not",
    "didn't": "did not",
    "i'm": "i am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "aren't": "are not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "isn't": "is not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "they're": "they are",
    "they've": "they have",
    "we're": "we are",
    "we've": "we have",
    "there's": "there is"
}

contractions_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')

def expand_contractions(text, pattern=contractions_pattern):
    def replace(match):
        return contractions_dict[match.group(0)]
    return pattern.sub(replace, text)

df_downsized['Review'] = df_downsized['Review'].apply(expand_contractions)

avg_length_after_cleaning = df_downsized['Review'].apply(len).mean()
print("Average length (in characters) after cleaning:", avg_length_after_cleaning)

# %% [markdown]
# # Pre-processing

# %%
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = " ".join(tokens)
    return processed_text
     
sample_indices = df_downsized.sample(3, random_state=42).index

print("SAMPLE REVIEWS BEFORE PREPROCESSING:")
for idx in sample_indices:
    print(f"Review {idx}:\n{df_downsized.loc[idx, 'Review']}")
    print("-"*80)   

avg_length_before_preprocessing = df_downsized['Review'].apply(len).mean()
print("Average length (in characters) before preprocessing:", avg_length_before_preprocessing)

df_downsized['Review'] = df_downsized['Review'].apply(preprocess_text)

print("SAMPLE REVIEWS AFTER PREPROCESSING:")
for idx in sample_indices:
    print(f"Review {idx}:\n{df_downsized.loc[idx, 'Review']}")
    print("-"*80)

avg_length_after_preprocessing = df_downsized['Review'].apply(len).mean()
print("Average length (in characters) after preprocessing:", avg_length_after_preprocessing)

print("Average length (in characters) before preprocessing:", avg_length_before_preprocessing)




# %% [markdown]
# # TF-IDF Feature Extraction

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  
    min_df=5,            
    max_df=0.8,          
    sublinear_tf=True
)

X = vectorizer.fit_transform(df_downsized['Review'])
y = df_downsized['Sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)



# %% [markdown]
# # Perceptron

# %%
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

perceptron_model = Perceptron()
perceptron_model.fit(X_train, y_train)

y_train_pred = perceptron_model.predict(X_train)
y_test_pred = perceptron_model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
prec_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

acc_test = accuracy_score(y_test, y_test_pred)
prec_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

print(acc_train)
print(prec_train)
print(recall_train)
print(f1_train)
print(acc_test)
print(prec_test)
print(recall_test)
print(f1_test)

# %% [markdown]
# # SVM

# %%
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

svm_model = SVC(kernel='linear', random_state=42)

svm_model.fit(X_train, y_train)

y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
prec_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

acc_test = accuracy_score(y_test, y_test_pred)
prec_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

print(acc_train)
print(prec_train)
print(recall_train)
print(f1_train)
print(acc_test)
print(prec_test)
print(recall_test)
print(f1_test)

# %% [markdown]
# # Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logreg_model = LogisticRegression(random_state=42)

logreg_model.fit(X_train, y_train)

y_train_pred = logreg_model.predict(X_train)
y_test_pred = logreg_model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
prec_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

acc_test = accuracy_score(y_test, y_test_pred)
prec_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

print(acc_train)
print(prec_train)
print(recall_train)
print(f1_train)
print(acc_test)
print(prec_test)
print(recall_test)
print(f1_test)

# %% [markdown]
# # Naive Bayes

# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_train_pred = nb_model.predict(X_train)

acc_train = accuracy_score(y_train, y_train_pred)
prec_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

acc_test = accuracy_score(y_test, y_test_pred)
prec_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

print(acc_train)
print(prec_train)
print(recall_train)
print(f1_train)

print(acc_test)
print(prec_test)
print(recall_test)
print(f1_test)


