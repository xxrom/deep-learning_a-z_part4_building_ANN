# Artificial Neural Network

# installing libraries

# Part 1 = Data Preprocessin

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding cadecorical data
# Страны перекодируем в числа и потом числа прекодируем в группы столбцов
# Пол человека перекодируем в число
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # France, Spain, Germany to 0, 1, 2
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Female, Male to 0, 1
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# разбиваем по столбцам только страны, чтобы убрать Dummy Variable Trap
# пол не нужно разбивать на два столбца (Dummy Variable Trap)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # убираем первый столбец из стран (Dummy Variable Trap)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense # инициализация слоев нейронки

# Initialising the ANN
classifier = Sequential() # инициализация пустого ANN classifier

# Adding the input layer and the first hidden layer
classifier.add(Dense(
  6, # hidden layer neurons (11 + 1) / 2
  input_dim = 11, # number of inputs (only in first layer)
  kernel_initializer = 'uniform', # init weights near zero
  activation = 'relu' # rectifier _/ function
))

# Adding the second hidden layer
classifier.add(Dense(
  6, # hiddent layer neurons (11 + 1) / 2
  kernel_initializer = 'uniform', # init weights near zero
  activation = 'relu' # rectifier _/ function
))

# Adding the output layer
classifier.add(Dense(
  1, # only 2 output (0 - stay and 1 - leave)
  kernel_initializer = 'uniform', # init weights near zero
  activation = 'sigmoid' # sigmoid S function (softmax if >2 outputs)
))

# Compiling the ANN
classifier.compile(
  optimizer = 'adam', # algorithm (gradient descent)
  # для оптимизации параметров (logarithmic loss)
  # если больше двух выходов cadecorical_crossentropy
  # например линейная модель - наименьшие расстояния от прямой до точек
  loss = 'binary_crossentropy',
  # алгоритм узнавания качества модели (как менять веса сети)
  metrics = ['accuracy']
)

# Fitting the ANN to the Training set
classifier.fit(
  X_train,
  y_train,
  batch_size = 10, # количество проходов перед обновлением весов
  nb_epoch = 100 # количество проходов по всем данным
) # get accuracy around 86%

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # переводим из % в True, False для cm
# если нужна более чуткая модель, то можно поставить 0.6

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# получил около 85.5% точность на тестовых данных (1710/2000)









