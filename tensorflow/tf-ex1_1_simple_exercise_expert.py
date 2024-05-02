#%% 1. 텐서플로 패키지 임포트
import tensorflow as tf

#%% 2. 데이터 지정
wo, bo = 2., 1.
def f(x): 
    return wo * x + bo
x = tf.constant([0.,1.])
y = f(x)

#%% 3. 인공신경망 모델링
# w와 b로 구성된 선형 모델
w = tf.Variable(1.0)
b = tf.Variable(0.0)

# 손실함수 정의
@tf.function
def loss(x, y):
    y_p = w * x + b
    return tf.reduce_mean((y - y_p)**2)

optimizer = tf.optimizers.SGD(0.1)

#%% 4. 주어진 데이터로 모델 학습
print("Initial: ", "w: {w.numpy()}, b: b.numpy()")
for e in range(100):
    with tf.GradientTape() as tape:
        l = loss(x, y)
    w_grad, b_grad = tape.gradient(l, [w, b])
    optimizer.apply_gradients(zip([w_grad, b_grad], [w, b]))
    
    if e % 10 == 0:
        print(f'{e}th-Trained, w_grad: {w_grad.numpy():.2}, b_grad: {b_grad.numpy():.2}, ',
              f'w: {w.numpy():.1}, b: {b.numpy():.1}')

#%% 5.성능 평가
x_test = tf.constant([2.,3.,4.])
y_test = f(x_test)
y_pred = w * x_test + b
errors = y_test - y_pred

disp = lambda v: [f'{e.numpy():.3}' for e in v]
print('Targets:', disp(y_test))
print('Predictions:', disp(y_pred))
print('Errors:', disp(errors))

# %%
