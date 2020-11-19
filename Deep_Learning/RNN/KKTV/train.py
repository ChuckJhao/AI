import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from sklearn import metrics
import time
import sys

#########################################################################

if sys.argv[1] == 'train' :
    data_name = 'feature.csv'
    drop_column = ['Unnamed: 0']
    reshape_value = 34
    '''
    #  0 ~ 27 : time slot 0 ~ 27
    # 28 : continuous week
    # 29 ~ 31 : Platform
    # 32 ~ 33 : connect type
    # 34 : watch ratio
    # 35 : hot drama
    # 36 : new episode next week
    '''
    shape_num = 38 - len(drop_column)
elif sys.argv[1] == 'train84' :
    data_name = 'time_feature.csv'
    drop_column = ['Unnamed: 0', '0']
    reshape_value = 35
    #Total 84 columns from 
    #1/7(day) 1:00 ~ 3:00, 1/7(day) 3:00 ~ 5:00, ..., 7/7(day) 23:00 ~ (another week) 1:00
    shape_num = 86 - len(drop_column)
elif sys.argv[1] == 'train168' :
    data_name = 'time_feature_ver2.csv'
    drop_column = ['Unnamed: 0']
    reshape_value = 34
    #Total 168 columns from
    #1/7(day) 0:00 ~ 1:00, 1/7(day) 1:00 ~ 2:00, ..., 7/7(day) 23:00 ~ 24:00
    shape_num = 169 - len(drop_column)
else :
    print('Please enter $python train.py <train/train84/train168> <save model(.h5)>')
    exit(0)

#########################################################################

start_time = time.asctime( time.localtime(time.time()) )

data = pd.read_csv(data_name)
print(data.shape)
data = data.drop(columns = drop_column)

data = data.to_numpy()
data = data.reshape(-1, reshape_value, shape_num)
n_data = []
for line in data :
    line = np.delete(line, -1, axis=0)
    line = np.delete(line, -1, axis=0)
    if sys.argv[1] == 'train84' :
        line = np.delete(line, -1, axis = 0)
    n_data.append(line)
n_data = np.array(n_data)
label = pd.read_csv('labels.csv')
label = label.drop(columns=['1', '2'])
label = label.to_numpy()

x_, x_test , y_, y_test = train_test_split(n_data, label, test_size=0.3, shuffle=True, random_state=110)
x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.3, shuffle=True, random_state=30)
'''
x_train = n_data[:35001, :, :]
x_test = n_data[35001: , :, :]
y_train = label[:35001, :]
y_test = label[35001: , :]
'''
x = sequence_input = Input(shape=(32, shape_num))

x = LSTM(128, dropout=0.375)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(28, activation='sigmoid')(x)

model = Model(sequence_input, preds)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, shuffle=True, epochs=100, batch_size=32, validation_data = (x_val, y_val))

model.save(sys.argv[2] + '.h5')

score = model.evaluate(n_data, label)

print('Test loss : ', score[0])
print('Test accuracy : ', score[1])


train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
y_predict = model.predict(n_data)
'''
train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)
y_predict = y_predict.reshape(-1,1)
label = label.reshape(-1,1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
'''
print('Train AUC : ', metrics.roc_auc_score(y_train, train_predict))
print('Test AUC : ', metrics.roc_auc_score(y_test, test_predict))
print('Total AUC : ', metrics.roc_auc_score(label, y_predict))

print(sys.argv[0]) 
print('Execute from ', start_time, 'to', time.asctime( time.localtime(time.time()) ))

