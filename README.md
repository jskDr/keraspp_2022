# 케라스로 구현하는 딥러닝, In Tensorflow 2
## keraspp_2022
Python Codes in My Book - Keras

## colab_py37_k28 폴드
파이썬 3.7, 텐서플로/케라스 2.8을 지원하는 구글 콜랩에서 실행하는 코드
- 파일명 끝에 -colab이 붙은 경우는 아래의 수정이 필요한 노트북임
```
from keras import dataset ==> from tensorflow.keras import dataset
```

## cpu_only 폴드
GPU가 있지만 cpu로 돌려야 하는 경우에 사용하는 코드 
- 노트북 앞에 아래 두 줄의 코드를 추가함
```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## 책표지
![책표지 - 케라스로 구현하는 딥러닝](https://user-images.githubusercontent.com/2564509/160268272-7bcff1d1-6eb2-420a-81cc-6e40f893d26c.jpg)
