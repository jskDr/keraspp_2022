#%% 1. 텐서플로 패키지 임포트
import tensorflow as tf

#%% 2. 데이터 지정
# x = numpy.array([0, 1, 2, 3, 4]) 
x = tf.constant([0.,1.,2.,3.,4.])
y = 2 * x + 1

#%% 3. 인공신경망 모델링 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
model.compile('SGD', 'mse')

#%% # 4. 주어진 데이터로 모델 학습
model.fit(x[:2], y[:2], epochs=1000, verbose=0)

#%% 5.성능 평가
y_pred = model.predict(x[2:]).flatten()
y_pred = tf.constant(y_pred)
print('Targets:', y[2:])
print('Predictions:', y_pred)
print('Errors:', y[2:] - y_pred)

# %%
