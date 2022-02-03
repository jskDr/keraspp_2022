# 9.3.1 Lambda 계층이란?
from keras.layers import Lambda, Input
from keras.models import Model

# 9.3.2 파이썬 lambda 기능 이용
def Lambda_with_lambda():
    x = Input((2,))
    y = Lambda(lambda x: x**2+2*x+1)(x)
    m = Model(x, y)

    yp = m.predict_on_batch(np.array([[1,2],[3,4]]))
    print(yp)

# 9.3.3 Lambda 계층 전용 함수 이용
def Lambda_function():
    def kproc(x):
        return x ** 2 + 2 * x + 1

    def kshape(input_shape):
        return input_shape

    x = Input((2,))
    y = Lambda(kproc, kshape)(x)
    m = Model(x, y)

    yp = m.predict_on_batch(np.array([[1,2],[3,4]]))
    print(yp)
    
# 9.3.4  백엔드 함수 이용
from keras import backend as K
def Backend_for_Lambda():
    def kproc_concat(x):    
        m = K.mean(x, axis=1, keepdims=True)
        d1 = K.abs(x - m)
        d2 = K.square(x - m)
        return K.concatenate([x, d1, d2], axis=1)

    def kshape_concat(input_shape):
        output_shape = list(input_shape)
        output_shape[1] *= 3
        return tuple(output_shape)

    x = Input((3,))
    y = Lambda(kproc_concat, kshape_concat)(x)
    m = Model(x, y)

    yp = m.predict_on_batch(np.array([[1, 2, 3], [3, 4, 8]]))
    print(yp)
    
# 9.3.5 엔진 전용 함수 이용    
import tensorflow as tf
def TF_for_Lamda():
    def kproc_concat(x):    
        m = tf.reduce_mean(x, axis=1, keepdims=True)
        d1 = tf.abs(x - m)
        d2 = tf.square(x - m)
        return tf.concat([x, d1, d2], axis=1)

    def kshape_concat(input_shape):
        output_shape = list(input_shape)
        output_shape[1] *= 3
        return tuple(output_shape)

    x = Input((3,))
    y = Lambda(kproc_concat, kshape_concat)(x)
    m = Model(x, y)

    yp = m.predict_on_batch(np.array([[1, 2, 3], [3, 4, 8]]))
    print(yp)
    
# 9.3.6 케라스2의 확장 기능 이용 
def No_Lambda_with_keras2():
    x = Input((2,))
    y = x**2+2*x+1
    m = Model(x, y)

    yp = m.predict_on_batch(np.array([[1,2],[3,4]]))
    print(yp)
    
    
def main():
    print('Lambda with lambda')
    Lambda_with_lambda()

    print('Lambda function')
    Lambda_function()

    print('Backend for Lambda')
    Backend_for_Lambda()

    print('TF for Lambda')
    TF_for_Lamda()

    print('Define-by-run approach in Keras2')
    No_Lambda_with_keras2()

    
main()