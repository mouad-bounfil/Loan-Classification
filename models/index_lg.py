import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/newuser/Desktop/gen-ai/revision-ai/Loan Classification.csv")

df.head()
df.info()
df.shape
df.describe()
