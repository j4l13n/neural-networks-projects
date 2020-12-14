import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

df = pd.read_csv('diabetes.csv')


for idx, col in enumerate(df.columns):
    ax = plt.subplot(3,3,idx+1)
    ax.yaxis.set_ticklabels([])
    sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel=False, kde_kws={'linestyle': '-', 'color': 'black', 'label': "No diabetes"})
    sns.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel=False,
                kde_kws={'linestyle': '--', 'color': 'black', 'label': "Diabetes"})

    ax.set_title(col)
# Hide the 9th subplot (bottom right) since there are. only 8 plots

plt.subplot(3,3,9).set_visible(False)

plt.show()


df.isnull().any()


df.describe()

print("Number of rows with 0 values for each variable")

for col in df.columns:
    missing_rows = df.loc[df[col]== 0].shape[0]
    print(col + ": " + str(missing_rows))

df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)


for col in df.columns:
    missing_rows = df.loc[df[col]== 0].shape[0]
    print(col + ": " + str(missing_rows))

df.head()

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())


df.head()

df.describe()

df_scaled = preprocessing.scale(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

df_scaled['Outcome'] = df['Outcome']
df = df_scaled

df.describe().loc[['mean', 'std','max'],].round(2).abs()

X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


model = Sequential()

model.add(Dense(units=4, activation='sigmoid', input_dim=3))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

sgd = optimizers.SGD(lr=1)

model.compile(loss='mean_squared_error', optimizer=sgd)

np.random.seed(9)

X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([[0],[1],[1],[0]])

model.fit(X, y, epochs=1500, verbose=False)






