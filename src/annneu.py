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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_pandas import DataFrameMapper
import os


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# # Importing the dataset
train = pd.read_csv('../resources/train.csv')

test_x = pd.read_csv('../resources/test.csv')

#Split x and y
train_y = train['SalePrice']
train_x = train.drop(['SalePrice'], axis=1)

test_id_col = test_x['Id'].values.tolist()

#Combine test and train
df_x = pd.concat([train_x, test_x])

#Remove id columns
df_x = df_x.drop(['Id'], axis=1)


def toNumerical(df):
    # Convert potential numerical values to numerical
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
        df[row[0]] = df.loc[:, row[0]].replace(row[1], list(range(0, len(row[1]))))
    return df


def toDummies(df):
    # Encode labels
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
        df[val] = labelEncoder.fit_transform(df[val])
        # df_x = OneHotEncoder(categorical_features=val).fit_transform(df_x).toArray()

    # Create dummies
    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    return df


def prepare_missing_values(df):
    # Remove columns with more than 60% missing value,
    df = df.dropna(axis='columns', how='any', thresh=df_x.shape[0] * 0.6, subset=None, inplace=False)

    # Fill missing values with column mean
    df = df.fillna(df_x.median())

    # Remove columns with all the same value
    df = df.drop(df_x.std()[(df_x.std() == 0)].index, axis=1)
    return df


def scale_data(df):

    from sklearn.preprocessing import StandardScaler
    from sklearn_pandas import DataFrameMapper

    mapper = DataFrameMapper([(df.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(df.copy(), 4)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    return df


df_x = toNumerical(df_x)
df_x = toDummies(df_x)
df_x = prepare_missing_values(df_x)
df_x = scale_data(df_x)

# #TODO Outliars eliminieren

train_prepared_x = df_x[:train.shape[0]]
test_prepared_x = df_x[train.shape[0]:]


corr_data = train_prepared_x.join(pd.DataFrame({'SalePrice': train_y}))

corrmat = corr_data.corr()
k = 10 #number of variables
cols = list(corrmat.nlargest(k, 'SalePrice').index)
cols = cols[1:]


ann_train_x = train_prepared_x[cols]
ann_test_x = test_prepared_x[cols]


seed = 7
np.random.seed(seed)
# split into 67% for train and 33% for test
ann_train_x, ann_valid_x, ann_train_y, ann_valid_y = train_test_split(ann_train_x, train_y, test_size=0.25, random_state=seed)


def create_model():
    # create model
    m = Sequential()
    m.add(Dense(10, input_dim=ann_train_x.shape[1], activation='relu'))
    m.add(Dense(30, activation='relu'))
    m.add(Dense(40, activation='relu'))
    m.add(Dense(1))
    # Compile model
    m.compile(optimizer ='adam', loss = 'mean_squared_error', metrics =[metrics.mae])
    return m


model = create_model()
model.summary()

history = model.fit(ann_train_x, ann_train_y, validation_data=(ann_valid_x, ann_valid_y), epochs=175, batch_size=30)


prediction = model.predict(ann_test_x)


submission = pd.DataFrame()
submission['Id'] = test_id_col
submission['SalePrice'] = prediction

submission.to_csv('submission.csv', index=False)