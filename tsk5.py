import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

from sklearn import metrics

from warnings import filterwarnings
filterwarnings('ignore')




#See first 10 rows
df_train.head(10)


print("Number of Rows:",df_train.shape[0])
print("Number of Features:",df_train.shape[1])


df_train.isnull().sum()


df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(),inplace=True)
