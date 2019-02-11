import os
import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from math import ceil
from pandas.plotting import scatter_matrix
import plotly

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Any results you write to the current directory are saved as output.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ThresholdedReLU
from keras import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Importing the dataset
train = pd.read_csv('../input/train.csv')
test_x = pd.read_csv('../input/test.csv')

y_column = 'SalePrice'

# Analyse distribution of SalePrices
sns.distplot(train[y_column], kde=False, color='b', hist_kws={'alpha': 0.9})
plt.show()

train.info()

# Analyse data types

categorical = train.select_dtypes(include=['object'])
print(categorical.head())


# Create dummies for object data
def to_dummies(df):
    label_encoder = LabelEncoder()

    objects = list(df.select_dtypes(include=['object']))
    for obj in objects:
        df[obj] = label_encoder.fit_transform(df[obj].fillna('NA'))
    return df
    # return pd.get_dummies(df, columns=objects, drop_first=True)


# categorical = to_dummies(categorical)


numerical = train.select_dtypes(exclude=['object'])
print(numerical.head())
corr_mat = numerical.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corr_mat, vmax=.8, square=True)

k = 10  # number of variables for heatmap
cols1 = corr_mat.nlargest(k, 'SalePrice')['SalePrice']
cols1index = cols1.index
cm = np.corrcoef(train[cols1index].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=list(cols1index),
            xticklabels=list(cols1index))
plt.show()

cols = corr_mat.nsmallest(9, 'SalePrice')['SalePrice'].index
bla = numerical[cols]
bla[y_column] = numerical[y_column]

cols = bla.corr().index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=list(cols),
            xticklabels=list(cols))
plt.show()


def plot(x, y):
    plt.scatter(x=x, y=y)
    plt.title(list(x), ' vs ', list(y))
    plt.ylabel(list(y))
    plt.xlabel(list(x))
    plt.show()


def plot_all(df, y):
    for x in list(df):
        plot(df[x], df)


plot_all(categorical, train['SalePrice'])
plot_all(numerical, train['SalePrice'])
# Outliers
# ax = plt.scatter(x=categorical['MSZoning'], y=train['SalePrice'])
# plt.show()

# ax = plt.scatter(x=train['TotalBsmtSF'], y=train['SalePrice'])
# plt.show()

# ax = plt.scatter(x=train['LotArea'], y=train['SalePrice'])
# plt.show()

# ax = plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# plt.show()

# ax = plt.scatter(x=train['1stFlrSF'], y=train['SalePrice'])
# plt.show()

# ax = plt.scatter(x=train['BsmtFinSF1'], y=train['SalePrice'])
# plt.show()

cols_with_na = train.isna().sum()
cols_with_na = cols_with_na[cols_with_na > 0]
print(cols_with_na.sort_values(ascending=False))

cols_with_na.sort_values(ascending=True).plot(kind='barh')
plt.title('Missing values')
plt.show()

roof = pd.DataFrame(data=dict(LotConfig=train['LotConfig']))
dummy_roof = pd.get_dummies(roof, columns=list(roof), drop_first=True)