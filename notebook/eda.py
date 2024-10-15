import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv("input/loan-approval-prediction/train.csv")

train.info()
# %%
train.head()
