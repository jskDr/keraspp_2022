# 9.4.1 관련 패키지 임포트
import keras
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras import initializers 
igu = initializers.get('glorot_uniform')
iz = initializers.get('zeros')

# 9.4.2 새로운 계층의 클래스 만들기
class SFC(Layer):
    # FC: Simplified fully connected layer
    def __init__(self, No, **kwargs):
        self.No = No
        super().__init__(**kwargs)

    def build(self, inshape):
        self.w = self.add_weight("w", (inshape[1], self.No),
                                 initializer=igu)
        self.b = self.add_weight("b", (self.No,),
                                 initializer=iz)        
        super().build(inshape)  

    def call(self, x):
        return K.dot(x, self.w) + self.b

    def compute_output_shape(self, inshape):
        return (inshape[0], self.No)

def main():
    # 9.4.3 사용 데이터 준비하기
    x = np.array([0, 1, 2, 3, 4]) 
    y = x * 2 + 1

    # 9.4.4 신규 계층이 포함된 모델링 만들기
    model = keras.models.Sequential()
    model.add(SFC(1, input_shape=(1,)))
    model.compile('SGD', 'mse')

    # 9.4.5 모델의 학습과 성능 평가
    model.fit(x[:2], y[:2], epochs=1000, verbose=0)
    print('Targets:',y[2:])
    print('Predictions:', model.predict(x[2:]).flatten())
    print('Errors:', y[2:] - model.predict(x[2:]).flatten())

if __name__ == '__main__':
    main()