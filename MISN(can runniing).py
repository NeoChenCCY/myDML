# 導入函式庫
import numpy as np  
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split


# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
##(X_train, y_train), (X_test, y_test) = mnist.load_data()


df = pd.read_csv("2022-train-v2.csv")
print(df)
##dataLabel = df['sensor_point5_i_value','sensor_point6_i_value','sensor_point7_i_value','sensor_point8_i_value','sensor_point9_i_value','sensor_point10_i_value']

dataLabel = df.filter(items=['sensor_point5_i_value','sensor_point6_i_value','sensor_point7_i_value','sensor_point8_i_value','sensor_point9_i_value','sensor_point10_i_value'])
print(dataLabel)
dataLabel.info()
pd.DataFrame(dataLabel).to_csv("dataLabel.csv")
datafeatrue = df.drop(dataLabel,axis=1)
print(datafeatrue)
datafeatrue.info
pd.DataFrame(datafeatrue).to_csv("datafeatrue.csv")


X_train, X_test, y_train, y_test = train_test_split(datafeatrue, dataLabel, random_state=42, shuffle=True)



# 建立簡單的線性執行的模型
model = Sequential()
# Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數
model.add(Dense(units=5, input_dim=635, kernel_initializer='normal', activation='relu')) 
# Add output layer
model.add(Dense(units=6, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000001000，即第7個值為 1
y_TrainOneHot = np_utils.to_categorical(y_train) 
y_TestOneHot = np_utils.to_categorical(y_test) 

# 將 training 的 input 資料轉為2維
"""
X_train_2D = X_train.reshape(535,1*535).astype('float32')  
X_test_2D = X_test.reshape(100, 1*100).astype('float32')  

x_Train_norm = X_train_2D/255
x_Test_norm = X_test_2D/255
"""

# 進行訓練, 訓練過程會存在 train_history 變數中
##rain_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)  
train_history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=10, batch_size=800, verbose=2)  

# 顯示訓練成果(分數)
scores = model.evaluate(X_test, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

# 預測(prediction)
##X = x_Test_norm[0:10,:]
##predictions = np.argmax(model.predict(X), axis=-1)
# get prediction result
##print(predictions)