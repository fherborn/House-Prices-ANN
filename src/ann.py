import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_pandas import DataFrameMapper

train = pd.read_csv('../resources/train.csv')
test_x = pd.read_csv('../resources/test.csv')

train_y = train['SalePrice']
train_x = train.drop(['SalePrice'],axis=1,inplace=False)
test_id_col = test_x['Id'].values.tolist()

df_x = pd.concat([train_x, test_x])

missing_values_bound = 0.80

total = df_x.isnull().sum().sort_values(ascending=False)

percent = (df_x.isnull().sum()/df_x.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

df_x = df_x.drop((missing_data[missing_data['Percent'] > missing_values_bound]).index,1)

df_x = df_x.drop("Id", 1)
df_x[df_x.FireplaceQu != df_x.FireplaceQu].Fireplaces.unique()
df_x['FireplaceQu']=train['FireplaceQu'].fillna('NF')
df_x['GarageType'].isnull().sum()
df_x['GarageCond'].isnull().sum()
df_x['GarageFinish'].isnull().sum()
df_x['GarageYrBlt'].isnull().sum()
df_x['GarageQual'].isnull().sum()

df_x['GarageType']=df_x['GarageType'].fillna('NG')
df_x['GarageCond']=df_x['GarageCond'].fillna('NG')
df_x['GarageFinish']=df_x['GarageFinish'].fillna('NG')
df_x['GarageYrBlt']=df_x['GarageYrBlt'].fillna('NG')
df_x['GarageQual']=df_x['GarageQual'].fillna('NG')
df_x['BsmtExposure']=df_x['BsmtExposure'].fillna('NB')
df_x['BsmtFinType2']=df_x['BsmtFinType2'].fillna('NB')
df_x['BsmtFinType1']=df_x['BsmtFinType1'].fillna('NB')
df_x['BsmtCond']=df_x['BsmtCond'].fillna('NB')
df_x['BsmtQual']=df_x['BsmtQual'].fillna('NB')
df_x['MasVnrType'] = df_x['MasVnrType'].fillna('none')
df_x.Electrical = df_x.Electrical.fillna('SBrkr')
df_x["LotAreaCut"] = pd.qcut(df_x.LotArea,10)
df_x['LotFrontage']=df_x.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df_x['LotFrontage']=df_x.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df_x.drop("LotAreaCut",axis=1,inplace=True)


#all_columns = train.columns.values
#non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
#                   "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
#                   "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
#                   "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
#                   "ScreenPorch","PoolArea", "MiscVal"]
#categorical = [value for value in all_columns if value not in non_categorical]

# %%javascript
# IPython.OutputArea.prototype._should_scroll = function(lines) {
#     return false;
# }

#for c in non_categorical:
#    plt.figure(figsize=(12,6))
#    plt.scatter(x=train[c], y=train.SalePrice)
#    plt.xlabel(c, fontsize=13)
#    plt.ylabel("SalePrice", fontsize=13)
    #plt.ylim(0,100)

#df_x.drop(train[(train["GrLivArea"]>4000)&(df_x["SalePrice"]<300000)].index,inplace=True)
#df_x.isnull().sum()[df_x.isnull().sum()>0]
#df_x["MasVnrArea"].fillna(0, inplace=True)
#df_x.isnull().sum()[df_x.isnull().sum()>0]
#df_x.columns
#df_x.head(20)


df_x = pd.get_dummies(df_x)
df_x = df_x.fillna(df_x.mean())

#imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#data = imp.fit_transform(data)

#data

# Log transformation
#data = np.log(data)
#labels = np.log(labels)

# Change -inf to 0 again
#data[data==-np.inf]=0

mapper = DataFrameMapper([(df_x.columns, RobustScaler())])
scaled_data = mapper.fit_transform(df_x.copy())
scaled_data_df = pd.DataFrame(scaled_data, index=df_x.index, columns=df_x.columns)

#scaler = StandardScaler()
#scaler = RobustScaler()
#data = scaler.fit_transform(scaled_features_df)
pca = PCA()
pca.fit(scaled_data)
#np.cumsum(pca.explained_variance_ratio_)[:20]

variance = 0.99

nr_components=np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance)
#nr_components

pca = PCA(n_components=nr_components)
dataPCA = pca.fit_transform(scaled_data)

train_x = dataPCA[:train.shape[0], :]
test_x = dataPCA[train.shape[0]:, :]















#df_train = pd.read_csv('../resources/train.csv', index_col=0)

#total = df_train.isnull().sum().sort_values(ascending=False)
#percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#df_train = df_train.fillna(df_train.mean())

#standardizing data
#saleprice_scaled = StandardScaler().fit_transform(train_y[np.newaxis])


#bivariate analysis saleprice/grlivarea
#var = 'GrLivArea'
#data = pd.concat([train_y, train_x[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#df_train = pd.read_csv('../resources/train.csv')

#cols = ['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
#df_train = df_train[cols]
# Create dummy values
#df_train = pd.get_dummies(df_train)
#filling NA's with the mean of the column:
#df_train = df_train.fillna(df_train.mean())
# Always standard scale the data before using NN
#scale = StandardScaler()
#X_train = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']]
#X_train = scale.fit_transform(X_train)
# Y is just the 'SalePrice' column
#y = df_train['SalePrice'].values
seed = 7
np.random.seed(seed)
# split into 67% for train and 33% for test
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.33, random_state=seed)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer ='adam', loss = 'mean_squared_error', metrics =[metrics.mae])
    return model



model = create_model()
model.summary()

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=150, batch_size=32)

#
# # summarize history for accuracy
# plt.plot(history.history['mean_absolute_error'])
# plt.plot(history.history['val_mean_absolute_error'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
#
# df_test = pd.read_csv('../resources/test.csv')
# cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
# df_test['GrLivArea'] = np.log1p(df_test['GrLivArea'])
# df_test = pd.get_dummies(df_test)
# df_test = df_test.fillna(df_test.mean())
# X_test = df_test[cols].values
# # Always standard scale the data before using NN
# scale = StandardScaler()
# X_test = scale.fit_transform(X_test)

prediction = model.predict(test_x)



submission = pd.DataFrame()
submission['Id'] = test_id_col
submission['SalePrice'] = prediction

submission.to_csv('submission.csv', index=False)