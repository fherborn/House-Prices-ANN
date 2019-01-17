import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_pandas import DataFrameMapper

train = pd.read_csv('../resources/train.csv')
#test = pd.read_csv('../resources/test.csv')



missing_values_bound = 0.80

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#  TODO replace mit mean
train = train.drop((missing_data[missing_data['Percent'] > missing_values_bound]).index,1)


train = train.drop("Id", 1)
train[train.FireplaceQu != train.FireplaceQu].Fireplaces.unique()
train['FireplaceQu']=train['FireplaceQu'].fillna('NF')
train['GarageType'].isnull().sum()
train['GarageCond'].isnull().sum()
train['GarageFinish'].isnull().sum()
train['GarageYrBlt'].isnull().sum()
train['GarageQual'].isnull().sum()
train['GarageType']=train['GarageType'].fillna('NG')
train['GarageCond']=train['GarageCond'].fillna('NG')
train['GarageFinish']=train['GarageFinish'].fillna('NG')
train['GarageYrBlt']=train['GarageYrBlt'].fillna('NG')
train['GarageQual']=train['GarageQual'].fillna('NG')
train['BsmtExposure']=train['BsmtExposure'].fillna('NB')
train['BsmtFinType2']=train['BsmtFinType2'].fillna('NB')
train['BsmtFinType1']=train['BsmtFinType1'].fillna('NB')
train['BsmtCond']=train['BsmtCond'].fillna('NB')
train['BsmtQual']=train['BsmtQual'].fillna('NB')
train['MasVnrType'] = train['MasVnrType'].fillna('none')
train.Electrical = train.Electrical.fillna('SBrkr')
train["LotAreaCut"] = pd.qcut(train.LotArea,10)
train['LotFrontage']=train.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train['LotFrontage']=train.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train.drop("LotAreaCut",axis=1,inplace=True)


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

train.drop(train[(train["GrLivArea"]>4000)&(train["SalePrice"]<300000)].index,inplace=True)
train.isnull().sum()[train.isnull().sum()>0]
train["MasVnrArea"].fillna(0, inplace=True)
train.isnull().sum()[train.isnull().sum()>0]
train.columns
train.head(20)


trainY = train['SalePrice']
#testY = test['SalePrice']
# TODO <- GET Train Y <<<----
train.drop(['SalePrice'],axis=1,inplace=True)


data_dummy = pd.get_dummies(train)

#imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#data = imp.fit_transform(data)

#data

# Log transformation
#data = np.log(data)
#labels = np.log(labels)

# Change -inf to 0 again
#data[data==-np.inf]=0

data_dummy.head(20)
mapper = DataFrameMapper([(data_dummy.columns, RobustScaler())])
scaled_data = mapper.fit_transform(data_dummy.copy())
scaled_data_df = pd.DataFrame(scaled_data, index=data_dummy.index, columns=data_dummy.columns)
scaled_data_df.head(20)
data = scaled_data

#scaler = StandardScaler()
#scaler = RobustScaler()
#data = scaler.fit_transform(scaled_features_df)
pca = PCA()
pca.fit(data)
np.cumsum(pca.explained_variance_ratio_)[:20]

variance = 0.99

nr_components=np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance)
nr_components

pca = PCA(n_components=nr_components)
dataPCA = pca.fit_transform(data)













































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
#
# # Importing the dataset
# #dataset = pd.read_csv('Churn_Modelling.csv')
# #X = dataset.iloc[:, 3:13].values
# #y = dataset.iloc[:, 13].values
#
# # Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#
# labelencoder_X_1 = LabelEncoder()
# X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# # labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categorical_features=[1])
# X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]
#
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
#
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# # Part 2 - Now let's make the ANN!
#
# # Importing the Keras libraries and packages
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#
# # Initialising the ANN
#classifier = Sequential()
#
# # Adding the input layer and the first hidden layer
#classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=4))
# # classifier.add(Dropout(p = 0.1))
#
# # Adding the second hidden layer
#classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
# # classifier.add(Dropout(p = 0.1))
#
# # Adding the output layer
#classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#
# # Compiling the ANN
#classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Fitting the ANN to the Training set
#classifier.fit(dataPCA, trainY, batch_size=10, epochs=100)
#
# # Part 3 - Making predictions and evaluating the model
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
#
# # Predicting a single new observation
# """Predict if the customer with the following informations will leave the bank:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40
# Tenure: 3
# Balance: 60000
# Number of Products: 2
# Has Credit Card: Yes
# Is Active Member: Yes
# Estimated Salary: 50000"""
# new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# new_prediction = (new_prediction > 0.5)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
#
# cm = confusion_matrix(y_test, y_pred)
#
# # Part 4 - Evaluating, Improving and Tuning the ANN
#
# # Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


# TODO check wieviele units per layer und welche Aktivierungsfunktion

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=4))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=dataPCA, y=trainY, cv=10, n_jobs=-1)
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
