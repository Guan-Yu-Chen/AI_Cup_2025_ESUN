import pandas as pd
import numpy as np

df = pd.read_csv("../src/acct_predict.csv")
random_state = 1
ratio = 0.1

np.random.seed(random_state)

mask = np.random.rand(len(df)) < ratio

df.loc[mask, 'label'] = 1


df.to_csv("random01.csv", index=False)