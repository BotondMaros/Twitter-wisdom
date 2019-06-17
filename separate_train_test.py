import pandas as pd
from sklearn.model_selection import train_test_split

new_df = pd.read_csv('text_and_binary.csv')
new_df['Text'] = new_df['Text'].astype('U')
new_df['Label'] = new_df['Label'].astype(int)
train, test = train_test_split(new_df, test_size=0.15)

print(train.head(5))
print(test.head(5))

train.to_csv('train.csv')
test.to_csv('test.csv')
