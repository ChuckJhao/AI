import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, KFold
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
    print('Please enter $python k_fold_train.py <train/train84/train168> <save model(.h5)> <k_fold_number>')
    exit(0)

#########################################################################

start_time = time.asctime( time.localtime(time.time()) )

data = pd.read_csv(data_name)
data = data.drop(columns = drop_column)
data = data.to_numpy()
data = data.reshape(-1, reshape_value, shape_num)
print(data.shape)
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
x_train, x_testing, y_train, y_testing = train_test_split(x_, y_, test_size=0.3, shuffle=True, random_state=30)
'''
x_train = n_data[:35001, :, :]
x_test = n_data[35001: , :, :]
y_train = label[:35001, :]
y_test = label[35001: , :]
'''
x = sequence_input = Input(shape=(32, shape_num)) # 34 wk x feature_num

x = LSTM(128, dropout=0.375)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(28, activation='sigmoid')(x)

kf = KFold(n_splits=int(sys.argv[3]), shuffle=True)
count = 0
score = []
train_auc = []
test_auc = []
total_auc = []
for train_index, test_index in kf.split(x_train) :
    x_1, x_2 = x_train[train_index], x_train[test_index]
    y_1, y_2 = y_train[train_index], y_train[test_index]

    model = Model(sequence_input, preds)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.fit(x_2, y_2, epochs=100, batch_size=32, validation_data=(x_1, y_1))
    predicting = model.predict(n_data)
    x_train_predicting = model.predict(x_train)
    x_test_predicting = model.predict(x_test)
    if count == 0 :
        y_predict = predicting
        x_train_predict = x_train_predicting
        x_test_predict = x_test_predicting
    else :
        y_predict += predicting
        x_train_predict += x_train_predicting
        x_test_predict += x_test_predicting
    count+=1
    score.append(model.evaluate(x_test, y_test))
    train_auc.append(metrics.roc_auc_score(y_train, x_train_predicting))
    test_auc.append(metrics.roc_auc_score(y_test, x_test_predicting))
    total_auc.append(metrics.roc_auc_score(label, predicting))
    model.save('{}_{}.h5'.format( sys.argv[2],count))
    model.reset_states()

for i in range(len(test_auc)):
    print(i ,'th Model : ', sep = '')
    print('\tTrain AUC : ', train_auc[i])
    print('\tTest AUC : ', test_auc[i])
    print('\tTotal AUC : ', total_auc[i])

y_predict /= 5
x_train_predict /= 5
x_test_predict /= 5
#y_predict[y_predict >= 0.5] = 1
#y_predict[y_predict < 0.5] = 0
y_predict = y_predict.reshape(-1,1)
label = label.reshape(-1,1)
fpr, tpr, threshold = metrics.roc_curve(label, y_predict)
print('-'*30)
print('Combine Train AUC : ', metrics.roc_auc_score(y_train, x_train_predict))
print('Combine Test  AUC : ', metrics.roc_auc_score(y_test, x_test_predict))
print('Combine Total AUC : ', metrics.auc(fpr, tpr))

print(start_time)
print('To')
print(time.asctime( time.localtime(time.time()) ))