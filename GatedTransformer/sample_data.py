import pandas as pd

df = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_train_data.csv')
sample_df = df.sample(frac=0.01 )
sample_df.to_csv('/data/malghaja/Bachelor_thesis/tsample.csv', index=False)
dff = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_valid_data.csv')
sample_dff = dff.sample(frac=0.01 )
sample_dff.to_csv('/data/malghaja/Bachelor_thesis/vsample.csv' , index=False)
dff = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_test_data.csv')   
sample_dfff = dff.sample(frac=0.1 )
sample_dfff.to_csv('test_sample.csv', index=False)
