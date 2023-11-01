import pandas as pd


dft = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_train_data.csv')
print(f'training subjects: {dft["person_id"].unique()}')
dftest = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_valid_data.csv')
print(f'test subjects: {dftest["person_id"].unique()}')
dfv = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_test_data.csv')
print(f'validation subjects: {dfv["person_id"].unique()}')
