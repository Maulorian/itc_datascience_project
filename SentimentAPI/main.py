import re

from flask import Flask, render_template, request
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import TFBertForSequenceClassification
from string import punctuation
import spacy
nlp = spacy.load('en_core_web_sm')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', from_tf=True)
tweet_model = TFBertForSequenceClassification.from_pretrained('./model.11-0.53.h5', config='./config.json')

app = Flask(__name__)


def tweet_gen(df):
    def g():
        for row in df.itertuples():
            text = row.text
            tokenized = tokenizer(row.text, max_length=280, padding='max_length', truncation=True)
            yield {k: np.array(tokenized[k]) for k in tokenized}

    return g


input_names = ['input_ids', 'token_type_ids', 'attention_mask']
data_types = ({k: tf.int32 for k in input_names})
data_shapes = ({k: tf.TensorShape([None]) for k in input_names})


def tokenize_tweet(tweet_df):
    apple_tweets_test = tf.data.Dataset.from_generator(
        tweet_gen(tweet_df),
        data_types, data_shapes
    ).batch(1)

    predictions = tweet_model.predict(apple_tweets_test)
    pred = np.argmax(predictions.logits[0])
    print(pred, predictions.logits[0])
    if pred == 0:
        return "Negative"
    elif pred == 1:
        return "Neutral"
    return "Positive!"


def clean_up_tweet(txt):
    # Remove mentions
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    # Remove hashtags
    txt = re.sub(r'#', '', txt)
    # Remove retweets:
    txt = re.sub(r'RT : ', '', txt)
    # Remove urls
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt).strip()
    return txt


def expand_contractions(phrase):
    phrase = re.sub(r"\`", "\'", phrase)

    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase.strip()


def remove_punctuation(tweet):
    tweet = re.sub(f"[{punctuation}]", "", tweet).strip()
    return tweet


def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text).strip()


def lemmatize_tweet(tweet):
    lemmatized_text = nlp(tweet)
    lemmatized_text_lst = [word.lemma_ for word in lemmatized_text]
    return " ".join(lemmatized_text_lst)


@app.route('/predict', methods=["POST"])
def predict():
    tweet = request.form.get("tweet")
    tweet = clean_up_tweet(tweet)
    tweet = expand_contractions(tweet)
    tweet = remove_punctuation(tweet)
    tweet = deEmojify(tweet)
    tweet = lemmatize_tweet(tweet)

    result = tokenize_tweet(pd.DataFrame([{"text": tweet}]))
    return render_template('index.html', data={"result": result, "tweet": tweet})


@app.route('/')
def home():
    return render_template('index.html', data={"result": "", "tweet": ""})


def main():
    app.run()


if __name__ == '__main__':
    main()
