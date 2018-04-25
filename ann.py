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
from keras.layers import Dropout # отключает нероны, исключает переобучение

# Initialising the ANN
classifier = Sequential() # инициализация пустого ANN classifier

# Adding the input layer and the first hidden layer with Dropout
classifier.add(Dense(
  6, # hidden layer neurons (11 + 1) / 2
  input_dim = 11, # number of inputs (only in first layer)
  kernel_initializer = 'uniform', # init weights near zero
  activation = 'relu' # rectifier _/ function
))
classifier.add(Dropout( # добавляется к только добавленному слою нейронки
  rate = 0.1 # 10% если этого мало, то берем +10% и тд
))

# Adding the second hidden layer
classifier.add(Dense(
  6, # hiddent layer neurons (11 + 1) / 2
  kernel_initializer = 'uniform', # init weights near zero
  activation = 'relu' # rectifier _/ function
))
classifier.add(Dropout( # добавляется к только добавленному слою нейронки
  rate = 0.1 # 10% если этого мало, то берем +10% и тд
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


# home Work
'''
Use our ANN model to predict if the customer with the following informations will leave the bank:

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
So should we say goodbye to that customer ?
'''

new_prediction = classifier.predict(
  sc.transform( # масштабируем новые данные
    np.array([ # искуственно создаем двуменый массив
      [ # просто сравниваем данные и заполняем их
        0.0, # добавляем .0 к элементу чтобы убрать ошибку
#Data with input dtype int64 was converted to float64 by StandardScaler
       0,
       600,
       1, # Mail
       40, 3, 60000, 2, 1, 1, 50000]
    ])
  )
)
new_prediction = (new_prediction > 0.5)


# Part 4 - Evaluationg, Improving
# более точная точность будет, мы адльше разделим X_test на 10 строк
# и каждую строку еще на 10 частей , из них мы возьмем 9 кусочков строки
# для обучения модели и 1 одну часть для проверки точности
# в итоге модель 10 раз обучится и 10 раз себя проверит
# будет 10 значений точностей модели и мы возьмем среднее значение
# которое и будет нашим итоговым значением точности
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier(): # Initialising the ANN скопировал выше строчки
  classifier = Sequential() # инициализация пустого ANN classifier
  classifier.add(Dense(
    6, # hidden layer neurons (11 + 1) / 2
    input_dim = 11, # number of inputs (only in first layer)
    kernel_initializer = 'uniform', # init weights near zero
    activation = 'relu' # rectifier _/ function
  ))
  classifier.add(Dense(
    6, # hiddent layer neurons (11 + 1) / 2
    kernel_initializer = 'uniform', # init weights near zero
    activation = 'relu' # rectifier _/ function
  ))
  classifier.add(Dense(
    1, # only 2 output (0 - stay and 1 - leave)
    kernel_initializer = 'uniform', # init weights near zero
    activation = 'sigmoid' # sigmoid S function (softmax if >2 outputs)
  ))
  classifier.compile(
    optimizer = 'adam', # algorithm (gradient descent)
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
  )
  return classifier

classifier = KerasClassifier(
  build_fn = build_classifier, # создаем новую ANN
  batch_size = 10,
  nb_epoch = 100
)

accuracies = cross_val_score(
  estimator = classifier,
  X = X_train,
  y = y_train,
  cv = 10 # на сколько кусков разбиваем X_train
  # , n_jobs = -1 # подключаем все ядра для выполнения
)
mean = accuracies.mean() # среднее значение получаем 79.6%
variance = accuracies.std() # среднее отклонение значений 1.01%

# Improving the ANN
# Dropout Regulatization to reduces overfitting if needed
# когда на Train данных модель выдает 90%, а Test выдает 70% (переобучение)



# Tuning the ANN




