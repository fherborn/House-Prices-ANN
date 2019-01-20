# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn_pandas import DataFrameMapper
#
# train = pd.read_csv('../resources/train.csv')
# #test = pd.read_csv('../resources/test.csv')
#
#
#
# missing_values_bound = 0.80
#
# total = train.isnull().sum().sort_values(ascending=False)
#
# percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
#
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# missing_data.head(20)
#
# #  TODO replace mit mean
# train = train.drop((missing_data[missing_data['Percent'] > missing_values_bound]).index,1)
#
#
# train = train.drop("Id", 1)
# train[train.FireplaceQu != train.FireplaceQu].Fireplaces.unique()
# train['FireplaceQu']=train['FireplaceQu'].fillna('NF')
# train['GarageType'].isnull().sum()
# train['GarageCond'].isnull().sum()
# train['GarageFinish'].isnull().sum()
# train['GarageYrBlt'].isnull().sum()
# train['GarageQual'].isnull().sum()
# train['GarageType']=train['GarageType'].fillna('NG')
# train['GarageCond']=train['GarageCond'].fillna('NG')
# train['GarageFinish']=train['GarageFinish'].fillna('NG')
# train['GarageYrBlt']=train['GarageYrBlt'].fillna('NG')
# train['GarageQual']=train['GarageQual'].fillna('NG')
# train['BsmtExposure']=train['BsmtExposure'].fillna('NB')
# train['BsmtFinType2']=train['BsmtFinType2'].fillna('NB')
# train['BsmtFinType1']=train['BsmtFinType1'].fillna('NB')
# train['BsmtCond']=train['BsmtCond'].fillna('NB')
# train['BsmtQual']=train['BsmtQual'].fillna('NB')
# train['MasVnrType'] = train['MasVnrType'].fillna('none')
# train.Electrical = train.Electrical.fillna('SBrkr')
# train["LotAreaCut"] = pd.qcut(train.LotArea,10)
# train['LotFrontage']=train.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# train['LotFrontage']=train.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# train.drop("LotAreaCut",axis=1,inplace=True)
#
#
# #all_columns = train.columns.values
# #non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
# #                   "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
# #                   "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
# #                   "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
# #                   "ScreenPorch","PoolArea", "MiscVal"]
# #categorical = [value for value in all_columns if value not in non_categorical]
#
# # %%javascript
# # IPython.OutputArea.prototype._should_scroll = function(lines) {
# #     return false;
# # }
#
# #for c in non_categorical:
# #    plt.figure(figsize=(12,6))
# #    plt.scatter(x=train[c], y=train.SalePrice)
# #    plt.xlabel(c, fontsize=13)
# #    plt.ylabel("SalePrice", fontsize=13)
#     #plt.ylim(0,100)
#
# train.drop(train[(train["GrLivArea"]>4000)&(train["SalePrice"]<300000)].index,inplace=True)
# train.isnull().sum()[train.isnull().sum()>0]
# train["MasVnrArea"].fillna(0, inplace=True)
# train.isnull().sum()[train.isnull().sum()>0]
# train.columns
# train.head(20)
#
#
# trainY = train['SalePrice']
# #testY = test['SalePrice']
# # TODO <- GET Train Y <<<----
# train.drop(['SalePrice'],axis=1,inplace=True)
#
#
# data_dummy = pd.get_dummies(train)
#
# #imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# #data = imp.fit_transform(data)
#
# #data
#
# # Log transformation
# #data = np.log(data)
# #labels = np.log(labels)
#
# # Change -inf to 0 again
# #data[data==-np.inf]=0
#
# data_dummy.head(20)
# mapper = DataFrameMapper([(data_dummy.columns, RobustScaler())])
# scaled_data = mapper.fit_transform(data_dummy.copy())
# scaled_data_df = pd.DataFrame(scaled_data, index=data_dummy.index, columns=data_dummy.columns)
# scaled_data_df.head(20)
# data = scaled_data
#
# #scaler = StandardScaler()
# #scaler = RobustScaler()
# #data = scaler.fit_transform(scaled_features_df)
# pca = PCA()
# pca.fit(data)
# np.cumsum(pca.explained_variance_ratio_)[:20]
#
# variance = 0.99
#
# nr_components=np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance)
# nr_components
#
# pca = PCA(n_components=nr_components)
# dataPCA = pca.fit_transform(data)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#



























# #---- ANN
#
# # Artificial Neural Network
#
# # Installing Theano
# # pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
#
# # Installing Tensorflow
# # pip install tensorflow
#
# # Installing Keras
# # pip install --upgrade keras
#
# # Part 1 - Data Preprocessing
#
# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# # Importing the dataset
train = pd.read_csv('../resources/train.csv')
#test = pd.read_csv('../resources/test.csv')

#dataset = pd.concat([train, test])
df_x = train.iloc[:, :-1]
df_y = train.iloc[:, -1]

#Remove id columns
df_x = df_x.drop(['Id'], axis=1)


#Convert potential numerical values to numerical
numerical = [
    ('ExterQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('ExterCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('HeatingQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('KitchenQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtQual', [np.nan, 'Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtCond', [np.nan, 'Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('PoolQC', [np.nan, 'Fa', 'TA', 'Gd', 'Ex']),
    ('GarageQual', [np.nan, 'Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('GarageCond', [np.nan, 'Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('FireplaceQu', [np.nan, 'Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtExposure', [np.nan, 'No', 'Mn', 'Av', 'Gd']),
    ('BsmtFinType1', [np.nan, 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
    ('BsmtFinType2', [np.nan, 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
    ('GarageFinish', [np.nan, 'Unf', 'RFn', 'Fin']),
]

for row in numerical:
    df_x[row[0]] = df_x.loc[:, row[0]].replace(row[1], list(range(0, len(row[1]))))


#Fill missing values with column mean
df_x = df_x.fillna(df_x.mean())

#Remove columns with all the same value
df_x = df_x.drop(df_x.std()[(df_x.std() == 0)].index, axis=1)


#Encode labels
categorical = [
    'MSZoning',
    'Street',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'Heating',
    'CentralAir',
    'Electrical',
    'Functional',
    'GarageType',
    'PavedDrive',
    'SaleType',
    'SaleCondition'
]

labelEncoder = LabelEncoder()

for val in categorical:
    df_x[val] = labelEncoder.fit_transform(df_x[val])

df_x = pd.get_dummies(df_x, columns=categorical)

#TODO create dummies
#TODO dummytrap vermeiden
#TODO Outliars eliminieren

#Remove columns with more than 60% missing value,
df_x = df_x.dropna(axis='columns', how='any', thresh=df_x.shape[0]*0.6, subset=None, inplace=False)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


# TODO check wieviele units per layer und welche Aktivierungsfunktion
def build_classifier():
    c = Sequential()
    c.add(Dense(units=100, activation='tanh', input_dim=train_x.shape[1]))
    c.add(Dense(units=50, activation='tanh'))
    c.add(Dense(units=20, activation='tanh'))
    c.add(Dense(units=1, activation='relu'))
    c.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return c


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=train_x, y=train_y, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()





# # Improving the ANN
# # Dropout Regularization to reduce overfitting if needed
#
# # Tuning the ANN
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
#
#
# def build_classifier(optimizer):
#     classifier = Sequential()
#     classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
#     classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#     classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return classifier
#
#
# classifier = KerasClassifier(build_fn=build_classifier)
# parameters = {'batch_size': [25, 32],
#               'epochs': [100, 500],
#               'optimizer': ['adam', 'rmsprop']}
# grid_search = GridSearchCV(estimator=classifier,
#                            param_grid=parameters,
#                            scoring='accuracy',
#                            cv=10)
# grid_search = grid_search.fit(X_train, y_train)
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_
