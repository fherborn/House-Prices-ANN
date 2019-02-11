
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot  as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Any results you write to the current directory are saved as output.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU, ELU, ThresholdedReLU, Softmax
from keras import metrics

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# Importing the dataset
train = pd.read_csv('../input/train.csv')
test_x = pd.read_csv('../input/test.csv')

# Remove outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

y_column = 'SalePrice'

# Split x and y
train_y = train[y_column]
train_x = train.drop([y_column], axis=1)

# Save id col from test set to append it at the end
test_id_col = test_x['Id'].values.tolist()

# Combine test and train set
df_x = pd.concat([train_x, test_x])

# Remove id columns
df_x = df_x.drop(['Id'], axis=1)

#convert potential numerical  categorical data to numerical data
qualityValues = ([np.nan, 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])
qualityCols = [
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'BsmtCond',
    'HeatingQC',
    'KitchenQual',
    'FireplaceQu',
    'GarageQual',
    'GarageCond',
    'PoolQC',
]

streetValues = ([np.nan, 'Pave', 'Grvl'], [0, 1, 2])
streetCols = [
    'Street',
    'Alley'
]

shapeValues = ([np.nan, 'IR3', 'IR2', 'IR1', 'Reg'], [0, 1, 2, 3, 4])
shapeCols = [
    'LotShape'
]

utilityValues = ([np.nan, 'ELO', 'NoSeWa', 'NoSewr', 'AllPub'], [0, 1, 2, 3, 4])
utilityCols = [
    'Utilities'
]

slopeValues = ([np.nan, 'Sev', 'Mod', 'Gtl'], [0, 1, 2, 3])
slopeCols = [
    'LandSlope'
]

exposureValues = ([np.nan, 'No', 'Mn', 'Av', 'Gd'], [0, 1, 2, 3, 4])
exposureCols = [
    'BsmtExposure'
]

finTypeValues = ([np.nan, 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],[0, 1, 2, 3, 4, 5, 6])
finTypeCols = [
    'BsmtFinType1',
    'BsmtFinType2',
]

finGarageValues = ([np.nan, 'Unf', 'RFn', 'Fin'],[0, 1, 2, 3])
finGarageCols = [
    'GarageFinish',
]

paveDriveValues = ([np.nan, 'N', 'P', 'Y'], [0, 1, 2, 3])
paveDriveCols = [
    'PavedDrive',
]

fenceQualValues = ([np.nan, 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], [0, 1, 2, 3, 4])
fenceQualCols = [
    'Fence',
]


def convert(df, cols, values):
    for col in cols:
        df[col] = df.loc[:, col].replace(values[0], values[1])
    return df


df_x = convert(df_x, qualityCols, qualityValues)
df_x = convert(df_x, streetCols, streetValues)
df_x = convert(df_x, shapeCols, shapeValues)
df_x = convert(df_x, utilityCols, utilityValues)
df_x = convert(df_x, slopeCols, slopeValues)
df_x = convert(df_x, exposureCols, exposureValues)
df_x = convert(df_x, finTypeCols, finTypeValues)
df_x = convert(df_x, finGarageCols, finGarageValues)
df_x = convert(df_x, paveDriveCols, paveDriveValues)
df_x = convert(df_x, fenceQualCols, fenceQualValues)


# Create dummies for object data
def to_dummies(df):
    label_encoder = LabelEncoder()

    objects = list(df.select_dtypes(include=['object']))
    for obj in objects:
        df[obj] = label_encoder.fit_transform(df[obj].fillna('NA'))

    return pd.get_dummies(df, columns=objects, drop_first=True)


df_x = to_dummies(df_x)


# Correct missing values
def prepare_missing_values(df):
    # Remove columns with more than 60% missing value,
    df = df.dropna(axis='columns', how='any', thresh=df_x.shape[0] * 0.6, subset=None, inplace=False)
    # Fill missing values with column median
    df = df.fillna(df_x.mean())
    # Remove columns with all the same value
    df = df.drop(df_x.std()[(df_x.std() == 0)].index, axis=1)
    return df


df_x = prepare_missing_values(df_x)


# Scale Data
def scale_data(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn_pandas import DataFrameMapper
    mapper = DataFrameMapper([(df.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(df.copy(), 4)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    return df


df_x = scale_data(df_x)



# Split train and test set
train_prepared_x = df_x[:train.shape[0]]
test_prepared_x = df_x[train.shape[0]:]




pca = PCA()
pca.fit(train_prepared_x, train_y)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

variance = 0.99

nr_components=np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance)


pca = PCA(n_components=nr_components)
pca.fit(train_prepared_x, train_y)

ann_train_x = pca.transform(train_prepared_x)
ann_test_x = pca.transform(test_prepared_x)

#ann_train_x = train_prepared_x
#ann_test_x = test_prepared_x

# Create correlation matrix
#corr_data = train_prepared_x
#corr_data[y_column] = train_y


# Drop all  columns wich correlation with Y is lower than 20%
#corr_mat = corr_data.corr().abs()
#y_corr = corr_mat[y_column].drop([y_column])
#filtered = corr_mat[corr_mat[y_column] > 0.6]
#potential_drop = filtered


# Drop all columns wich correlation with other columns is higher than 95%
# Select upper triangle of correlation matrix
#upper = potential_drop.where(np.triu(np.ones(potential_drop.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
#to_drop = list([potential_drop for potential_drop in upper.columns if any(upper[potential_drop] > 0.95)])
#cols = list(filter(lambda x: x not in to_drop, filtered))
#cols = list(filtered)


# Select columns for network
#ann_train_x = train_prepared_x[cols]
#ann_test_x = test_prepared_x[cols]

#ann_train_x = train_prepared_x
#ann_test_x = test_prepared_x


# Set seed for randomization
seed = 7
np.random.seed(seed)

# Split train set into train and valid
#ann_train_x, ann_valid_x, ann_train_y, ann_valid_y = train_test_split(ann_train_x, train_y, test_size=0.25, random_state=seed)

# Create model
input_layer_nodes = ann_train_x.shape[1]
output_layer_nodes = 1
hidden_layer_nodes = int(ann_train_x.shape[0]/(1*(input_layer_nodes + output_layer_nodes)))


def create_model():
    m = Sequential()
    m.add(Dense(100, input_dim=input_layer_nodes, activation='relu'))
    m.add(Dense(50, activation='relu'))
    #m.add(Dense(hidden_layer_nodes, input_dim=input_layer_nodes, activation=LeakyReLU(alpha=0.1)))
    m.add(Dense(output_layer_nodes))
    m.compile(optimizer=keras.optimizers.Adadelta(), loss = 'mean_squared_error', metrics=[metrics.mse])
    return m

#import random
#from sklearn.ensemble import RandomForestRegressor
#parameters = dict(bootstrap= True,
#              min_samples_leaf= 3,
#              n_estimators=  10,
#              min_samples_split= 10,
#              max_features= 'auto',
#              max_depth= 30,
#              max_leaf_nodes= None)
#
#random.seed(42)
#rf = RandomForestRegressor(**parameters)
#rf.fit(train_prepared_x, train_y)

model = create_model()
model.summary()

#from sklearn.model_selection import GridSearchCV

#parameters = dict(batch_size =  [16, 25, 32], epochs=[100, 500, 1000], optimizer= [keras.optimizers.Adadelta(), 'adam', 'rmsprop'])
#grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=2)

#grid_search = grid_search.fit(ann_train_x, train_y)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_

#print('Best Params: ', best_parameters)

history = model.fit(ann_train_x, train_y, epochs=100, batch_size=32)


#pred_df = pd.DataFrame()
#pred_df['Predicted']  = list(model.predict(ann_valid_x))
#pred_df['Real']  = ann_valid_y

#print(pred_df)


prediction = model.predict(ann_test_x)


submission = pd.DataFrame()
submission['Id'] = test_id_col
submission['SalePrice'] = prediction
submission.to_csv('submission.csv', index=False)


target_submission = pd.read_csv('../input/good_submission.csv')

compare = submission
compare['Target'] = target_submission[y_column]
compare['Diff'] = submission[y_column]-target_submission[y_column]

print(compare)


def rmsle(ypred, ytest) :
    assert len(ytest) == len(ypred)
    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest))**2))


print('Error: ', rmsle(submission[y_column].tolist(), target_submission[y_column].tolist()))

print('Finished :-)')
