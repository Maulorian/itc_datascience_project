import time

import pandas as pd
import tweepy as tweepy

LIMIT = 10

API_KEY = 'tGmVjH9XLvI9t28ZlZeQuzT2W'
API_KEY_SECRET = 'M20f5SNWJwInKeX1Xb7Mfe0ZipMjxUXLISjuk1VLmNQvzitHIh'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAADR3MgEAAAAAhE0oyEMHM5NIEF3Quzie3KrDiFw%3Doq9VxmYzsee7D6CPnIIjO7YpHJcYwJJuIG4GfUPVj8h3emCbx9'
ACCESS_TOKEN = '941735938962722821-eLPVVZD0aKMNB1IbZzFNzvzFsKtUZKY'
ACESSS_TOKEN_SECRET = 'gLZnMd2g2NcO2QgSSJWMk9uylhPdJrBOj0mqq5jMLt4ey'

auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACESSS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)


def scrape_tweets(query):
    try:
        # Creation of query method using parameters
        tweets = tweepy.Cursor(api.search, q=query, lang='en').items(LIMIT)

        # Pulling information from tweets iterable object
        tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]

        # Creation of dataframe from tweets list
        # Add or remove columns as you remove tweet information
        return pd.DataFrame(tweets_list, columns=['created_at', 'id', 'text'])
    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)


def limit_handled(cursor):
    while True:
        try:
            yield next(cursor)
        except tweepy.RateLimitError:
            time.sleep(15 * 60)


def main():
    companies = pd.read_csv('companies.csv')
    companies = [c for c in companies.itertuples()]
    for company in companies:
        print(company.Name)
        tweets = scrape_tweets(query=company.Name)
        tweets['company_name'] = company.Name
        tweets['company_sector'] = company.Sector
        tweets['text'] = tweets['text'].str.replace('\n', '')
        tweets.to_csv('tweets.csv', mode='a', header=False, index=False)


if __name__ == '__main__':
    main()
