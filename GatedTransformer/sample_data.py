import pandas as pd

df = pd.read_csv('data.csv')
sample_df = df.sample(frac=0.01 )
sample_df.to_csv('sample.csv', index=False)