
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# # Importing the dataset
train = pd.read_csv('../resources/train.csv')
test_x = pd.read_csv('../resources/test.csv')

#Split x and y
train_y = train['SalePrice']
train_x = train.drop(['SalePrice'], axis=1)

#Combine test and train
df_x = pd.concat([train_x, test_x])

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

#for val in categorical:
#    df_x[val] = labelEncoder.fit_transform(df_x[val])
    #df_x = OneHotEncoder(categorical_features=val).fit_transform(df_x).toArray()

#TODO remove
df_x = df_x.drop(categorical, axis=1)
#TODO remove end

#Create dummies
#df_x = pd.get_dummies(df_x, columns=categorical, drop_first=True)

#TODO Outliars eliminieren

#Remove columns with more than 60% missing value,
df_x = df_x.dropna(axis='columns', how='any', thresh=df_x.shape[0]*0.6, subset=None, inplace=False)

#Fill missing values with column mean
#TODO later? after categorical
df_x = df_x.fillna(df_x.median())

#Remove columns with all the same value
#TODO later? after categorical
df_x = df_x.drop(df_x.std()[(df_x.std() == 0)].index, axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

#split test and train
train_x = df_x[:train_x.shape[0]].values
test_x = df_x[test_x.shape[0]:].values

#TODO eventuell Ueberfluessig? somit mehr daten  zum trainieren?
#train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
sc = StandardScaler()
#sc = MinMaxScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.fit_transform(test_x)
#valid_x = sc.transform(valid_x)


from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import metrics

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# TODO check wieviele units per layer und welche Aktivierungsfunktion
def build_classifier():
    c = Sequential()
    c.add(Dense(units=10, kernel_initializer='normal', activation='relu', input_dim=train_x.shape[1]))
    c.add(Dense(units=30, kernel_initializer='normal', activation='relu'))
    c.add(Dense(units=40, kernel_initializer='normal', activation='relu'))
    c.add(Dense(units=1))
    c.compile(loss='mean_squared_error', optimizer='adam', metrics =[metrics.mae])
    print(c.summary())
    return c



#classifier = build_classifier()
#classifier.fit(np.array(train_x), np.array(train_y), batch_size=25, epochs=300)

#pred_y = classifier.predict(test_x)

#pred_y = classifier.predict(test_x)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(valid_y, pred_y)

from sklearn.model_selection import KFold

#model = KerasRegressor(build_fn=build_classifier, batch_size=10, epochs=200)
model = build_classifier()
model.fit(train_x, train_y, batch_size=20, epochs=200)
#classifier.fit(train_x, train_y)
#kfold = KFold(n_splits=10, random_state=seed)
#accuracies = cross_val_score(estimator=model, X=train_x, y=train_y, cv=kfold, n_jobs=-1)
pred_y = model.predict(test_x)
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(valid_y, pred_y)

#import matplotlib.pyplot as plt
#plt.plot(cm[0], cm[1])
#plt.show()



# # Improving the ANN
#TODO dropout

# # Tuning the ANN
#from sklearn.model_selection import GridSearchCV

#classifier = KerasClassifier(build_fn=build_classifier)
#parameters = dict(batch_size=[10, 25, 32], epochs=[100, 200, 500])#, optimizer=['adam', 'rmsprop'])
#grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
#grid_search = grid_search.fit(train_x, train_y)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_
