import pandas as pd
import numpy as np 
import time
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('feature3.csv')
#data = data.drop(columns = ['Unnamed: 0', '0'])
data = data.drop(columns = ['Unnamed: 0', '0', '0.0.28', '0.0.29', '0.0.30', '0.0.31', '0.0.32', '0.0.33', '0.0.34', '0.0.35', '0.0.36'])
data = data.to_numpy()
data = data.reshape(-1, 34, 28)
print(data.shape)
n_data = []
for line in data :
    line = np.delete(line, -1, axis=0)
    line = np.delete(line, -1, axis=0)
    n_data.append(line)
n_data = np.array(n_data)
n_data = n_data.reshape(-1, 32*28)
label = pd.read_csv('labels.csv')
label = label.drop(columns=['1', '2'])
label = label.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(n_data, label, test_size=0.3, shuffle=True, random_state = 110)

model = LinearRegression()

model.fit(x_train, y_train)

train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
y_predict = model.predict(n_data)

print('Train AUC : ', metrics.roc_auc_score(y_train, train_predict))
print('Test AUC : ', metrics.roc_auc_score(y_test, test_predict))
print('Total AUC : ', metrics.roc_auc_score(label, y_predict))
