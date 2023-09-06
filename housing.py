import numpy as np   #Linear algera Library
import pandas as pd
import matplotlib.pyplot as plt  #to plot graphs
import seaborn as sns  #to plot graphs
from sklearn.linear_model import LinearRegression   #for linear regression model
sns.set()  #setting seaborn as default 

import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('Housing.csv')   #reads the input data
data.head()   #displays the first five rows

data.info()

data.describe(include ='all')   #parameter include=all will display NaN values as well

data.isnull().sum() # No null values

#first fetch all the categorical columns with Yes and NO
categorical =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
#write a function to change yes to 1 and no to 0
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# now replace yes and no with 1 and 0 in our dataset
data[categorical] = data[categorical].apply(binary_map)

data.head()

table = pd.get_dummies(data['furnishingstatus'])   #add the column into table variable
table.head()

table = pd.get_dummies(data['furnishingstatus'], drop_first = True)  #recreate table but now drop the first column(furnished)
table.head()

data = pd.concat([data, table], axis = 1)  #attach the other two columns to our data set
data.head()

data.drop(['furnishingstatus'], axis = 1, inplace = True) #drop the old column from the dataset
data.head()
data.columns

from sklearn.model_selection import train_test_split
np.random.seed(0) #so data can have same values
df_train, df_test = train_test_split(data, train_size = 0.7, test_size = 0.3, random_state = 100)

df_train.head()

from sklearn.preprocessing import MinMaxScaler    #to make all the numbers to the same scale
scaler = MinMaxScaler()

var_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
#appied scale to all numerical columns(not the yes/no and dummy columns)

#apply the scaled values to our training data set
df_train[var_to_scale] = scaler.fit_transform(df_train[var_to_scale])  

df_train.head()

df_train.describe()

# only output price is poped out of df_Train and put into y_train
y_train = df_train.pop('price') 
x_train = df_train

y_train.head()

#using linear regression
lm=LinearRegression()
lm.fit(x_train,y_train)
lm.coef_

#values from 0 to 1
#0 model explain None of the variability
#1 model explain Entire of the variability
lm.score(x_train,y_train)

var_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_test[var_to_scale] = scaler.fit_transform(df_test[var_to_scale])

y_test = df_test.pop('price')
x_test = df_test

#predict the output(predictions) using the test data
predictions = lm.predict(x_test)

from sklearn.metrics import r2_score 
r2_score(y_test, predictions)

#AttributeError: 'Series' object has no attribute 'flatten' --to avoid this error in the next step
y_test.shape
y_test_matrix = y_test.values.reshape(-1,1)

#load actual and predecited values side by side
dframe=pd.DataFrame({'actual':y_test_matrix.flatten(),'Predicted':predictions.flatten()}) 
#flatten toget single axis of data (1 dimension only)

dframe.head(15)

#using scatter plot compare the actual and predicted data
fig = plt.figure()
plt.scatter(y_test,predictions)
plt.title('Actual versus Prediction ')
plt.xlabel('Actual', fontsize=20)                         
plt.ylabel('Predicted', fontsize=20)  

#trying the same with a reg plot(optonal)
sns.regplot(x=y_test, y=predictions)
plt.title('Actual versus Prediction ')
plt.xlabel('Actual', fontsize=20)                         
plt.ylabel('Predicted', fontsize=20)  

from sklearn.metrics import mean_squared_error

rms = mean_squared_error(y_test, predictions, squared=False)
print(rms)
