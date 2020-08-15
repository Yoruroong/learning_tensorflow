import tensorflow as tf
import pandas as pd
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

try:
    with tf.device('/GPU:0'):
        ###########################
        # 데이터를 준비합니다.
        파일경로 = 'lemonade.csv'
        레모네이드 = pd.read_csv(파일경로)
        레모네이드.head()
        # 종속변수, 독립변수
        독립 = 레모네이드[['temp']]
        종속 = 레모네이드[['sell']]
        print(독립.shape, 종속.shape)
 
        ###########################
        # 모델을 만듭니다.
        X = tf.keras.layers.Input(shape=[1])
        Y = tf.keras.layers.Dense(1)(X)
        model = tf.keras.models.Model(X, Y)
        model.compile(loss='mse')
 
        ###########################
        # 모델을 학습시킵니다. 
        model.fit(독립, 종속, epochs=1000, verbose=0)
        model.fit(독립, 종속, epochs=100000000)
 
        ###########################
        # 모델을 이용합니다. 
        print(model.predict(독립))
        print(model.predict([[20200814]]))
except RuntimeError as e:
  print(e)
