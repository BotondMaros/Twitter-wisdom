import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from bert_text import run_on_dfs

raw = pd.read_csv('preprocessed_data.csv', encoding = 'utf8')
new_df = raw[['Text','Label']].copy()

rows = new_df.shape[0]
for i in range(rows):
  if new_df['Label'][i] == 'DY' or new_df['Label'][i]== 'PY':
    new_df['Label'][i] = 1
  else: new_df['Label'][i] = 0
new_df.to_csv('text_and_binary.csv')

new_df['Text'] = new_df['Text'].astype(str)
new_df['Label'] = new_df['Label'].astype(int)
train, test = train_test_split(new_df, test_size=0.1)

myparam = {
    "DATA_COLUMN": "Text",
    "LABEL_COLUMN": "Label",
    "LEARNING_RATE": 3e-5,
    "NUM_TRAIN_EPOCHS": 8
}

tf.logging.set_verbosity(tf.logging.INFO)
result, estimator = run_on_dfs(train, test, **myparam)
print(result)
'''
3 epochs
{'auc': 0.7481781, 'eval_accuracy': 0.7728813, 'f1_score': 0.6763284, 'false_negatives': 36.0, 'false_positives': 31.0, 'loss': 0.47683507, 'precision': 0.6930693, 'recall': 0.6603774, 'true_negatives': 158.0, 'true_positives': 70.0, 'global_step': 248}

4 epochs
{'auc': 0.7300237, 'eval_accuracy': 0.7457627, 'f1_score': 0.796748, 'false_negatives': 39.0, 'false_positives': 36.0, 'loss': 0.7819128, 'precision': 0.8032787, 'recall': 0.7903226, 'true_negatives': 73.0, 'true_positives': 147.0, 'global_step': 331}

5 epochs
{'auc': 0.7027134, 'eval_accuracy': 0.73220336, 'f1_score': 0.6183574, 'false_negatives': 44.0, 'false_positives': 35.0, 'loss': 0.77990496, 'precision': 0.64646465, 'recall': 0.5925926, 'true_negatives': 152.0, 'true_positives': 64.0, 'global_step': 414}
'''