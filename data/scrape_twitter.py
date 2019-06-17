import tweepy 
import pandas as pd
import numpy as np
import csv


#Botond's twitter api keys
consumer_key = "JYUC1WTudFvPbWWxGnwLdJlE1" 
consumer_secret = "LBa1RrIEm6EnbeYSiS5UxUHqsjQ8ny24oiWBZiS9sOIA77EPLu"
access_key = "2881734419-Frjwzikm7853LLksRMWGQ28rGnubhhsiixR9Met"
access_secret = "vVJq2qqAZByF00GdZsUPxDhufiayr7AtLDhoO0R7MfLVg"


#reading in the databases of the labeled data
train_with_metadata = "./train_with_metadata.csv"
test_with_metadata = "./test_with_metadata.csv"
dev_with_metadata = "./dev_with_metadata.csv"

train_df = pd.read_csv(train_with_metadata, names = ["ID", "Metadata", "Label"])
test_df = pd.read_csv(test_with_metadata, names = ["ID", "Metadata", "Label"])
dev_df = pd.read_csv(dev_with_metadata, names = ["ID", "Metadata", "Label"])

def get_new_list(df):
    # Authorization to consumer key and consumer secret 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    # Access to user's access key and access secret 
    auth.set_access_token(access_key, access_secret) 
    # Calling api 
    api = tweepy.API(auth)

    #create a list from the IDs
    id_list = list(df['ID'])
    ##create chunks of 100 for twitter API
    temp_id = [id_list[i:i + 100] for i in range(0, len(id_list), 100)]
    temp = []
    for i in range(0, len(temp_id)):
        tweets = api.statuses_lookup(temp_id[i])
        tweets_for_csv = [[int(tweet.id),tweet.text] for tweet in tweets]
        temp.append(tweets_for_csv)
    return temp


def df_merge(original_df, new_df):
    final_df = original_df.merge(new_df,how='left', left_on='ID', right_on='ID')
    final_df = final_df.drop_duplicates()
    final_df = final_df.dropna()
    final_df = final_df[['ID', "Metadata","Text","Label"]]
    return final_df


if __name__ == '__main__': 
    train_tweets = get_new_list(train_df)
    test_tweets = get_new_list(test_df)
    dev_tweets = get_new_list(dev_df)
    ##list have 25 sublist, contaning a sublist for each entity with id and string
    #have to flatten that out
    train_tweets_flat = [item for sublist in train_tweets for item in sublist]
    test_tweets_flat = [item for sublist in test_tweets for item in sublist]
    dev_tweets_flat = [item for sublist in dev_tweets for item in sublist]

    # with open('train_temp.csv', 'w') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(train_tweets_flat)
    # with open('test_temp.csv', 'w') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(test_tweets_flat)
    # with open('dev_temp.csv', 'w') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(dev_tweets_flat)

    labels = ["ID", "Text"]
    train_tweets_df = pd.DataFrame.from_records(train_tweets_flat, columns=labels)
    test_tweets_df = pd.DataFrame.from_records(test_tweets_flat, columns=labels)
    dev_tweets_df = pd.DataFrame.from_records(dev_tweets_flat, columns=labels)

    train_merged = df_merge(train_df,train_tweets_df)
    test_merged = df_merge(test_df,test_tweets_df)
    dev_merged = df_merge(dev_df,dev_tweets_df)

    complete_df = train_merged.append(test_merged)
    complete_df = complete_df.append(dev_merged)

    train_merged.to_csv("train_with_text.csv", encoding='utf-8', index=False)
    test_merged.to_csv("test_with_text.csv", encoding='utf-8', index=False)
    dev_merged.to_csv("dev_with_text.csv", encoding='utf-8', index=False)
    complete_df.to_csv("complete_dataset.csv", encoding='utf-8', index=False)
 

   