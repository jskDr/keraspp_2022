#%% 1. 텐서플로 패키지 임포트
import tensorflow as tf
import numpy as np

#%% 2. 데이터 지정
# x = numpy.array([0, 1, 2, 3, 4]) 
x = np.array([0.,1.,2.,3.,4.])
y = 2 * x + 1

#%% 3. 인공신경망 모델링 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
# model.compile('SGD', 'mse')

#%% # 4. 주어진 데이터로 모델 학습
# model.fit(x[:2], y[:2], epochs=1000, verbose=0)
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.MeanSquaredError(name="train_loss")
train_acc = tf.keras.metrics.R2Score(name="train_acc")
test_loss = tf.keras.metrics.MeanSquaredError(name="test_loss")
test_acc = tf.keras.metrics.R2Score(name="test_acc")

data = tf.constant(y[:2])
labels = tf.constant(x[:2])

# predictions = model(data, training=True)
# loss = loss_object(labels, predictions)

# w, b = model.weights
# print(f"W={w}, b={b}")
# print(predictions)
# print(loss)

#%% train
@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_acc(labels, predictions)

train_step(data, labels)

#%% 5.성능 평가
y_pred = model.predict(x[2:]).flatten()
y_pred = tf.constant(y_pred)


print('Targets:', y[2:])
print('Predictions:', y_pred)
print('Errors:', y[2:] - y_pred)

# %%
