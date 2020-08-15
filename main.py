import tensorflow as tf
import pandas as pd

###########################
# 데이터를 준비합니다.
파일경로 = 'AMD.csv'
주식 = pd.read_csv(파일경로)
주식.head()
# 종속변수, 독립변수
독립 = 주식[['date']]
종속 = 주식[['price']]
print(독립.shape, 종속.shape)
###########################
# 모델을 만듭니다.
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')
###########################
# 모델을 학습시킵니다. 
model.fit(독립, 종속, epochs=30000, verbose=0)
model.fit(독립, 종속, epochs=100000)
 
###########################
# 모델을 이용합니다. 
print(model.predict(독립))
print(model.predict([[20200814]]))
