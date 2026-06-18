---
title: "7. KNN_수박_참외_맞추기"
permalink: /study/class/7._KNN_수박_참외_맞추기_공유용/
date: 2026-06-18
---

# Google Colal 데이터 로드

```python
#Step 1. 구글 코랩에 한글 폰트 설정하기

!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

```python
#Step 1.분석할 데이터가 저장된 파일을 불러와서 변수에 할당합니다.
from google.colab import files
myfile = files.upload()
import io
import pandas as pd
#pd.read_csv로 csv파일 불러오기
src_data = pd.read_csv(io.BytesIO(myfile['수박과참외.csv']),
                       encoding='cp949')
src_data
```

# 로컬 데이터 로드

```python
#컴퓨터에서 작업하려면 아래 코드의 주석을 제거하고 실행하면 됩니다
import pandas as pd
src_data = pd.read_csv('./머신러닝실습용자료/수박과참외.csv',encoding='cp949')
src_data
```

# 공통 실습 코드

```python
#수박과 참외의 무게와 길이
수박정보 = src_data.loc[ (src_data['종류'] =='수박'), ['무게','길이']]
참외정보 = src_data.loc[ (src_data['종류'] =='참외'), ['무게','길이']]

import matplotlib.pyplot as plt
plt.scatter(수박정보.무게,수박정보.길이)
plt.scatter(참외정보.길이,참외정보.길이)
plt.xlabel('weigth')
plt.ylabel('length')
plt.show()
```

![7._KNN_수박_참외_맞추기_공유용_img1.png](/assets/img/study/7._KNN_수박_참외_맞추기_공유용_img1.png)

```python
import numpy as np

# np.column_stack을 통해 무게와 길이를 data 변수에 넣는다. 
data = np.column_stack((src_data.무게, src_data.길이))

# 데이터의 종류를 target에 넣는다.
target = src_data.종류

print(data)
print(target)
```

```
[[2000.    30. ]
 [2500.    25. ]
 [1800.    20. ]
 [1500.    16. ]
 [ 900.    10. ]
 [2500.    33. ]
 [2250.    23. ]
 [1860.    17. ]
 [2100.    21. ]
 [1500.    17. ]
 [ 500.     8. ]
 [ 400.     7.5]
 [ 450.     5. ]
 [ 400.     4.5]
 [ 600.     8.5]]
0     수박
1     수박
2     수박
3     수박
4     수박
5     수박
6     수박
7     수박
8     수박
9     수박
10    참외
11    참외
12    참외
13    참외
14    참외
Name: 종류, dtype: str

```

```python
# Step 4. 주어진 데이터를 훈련용과 테스트(검증용)으로 나눕니다. 

#test_size = 0.25, random_state = 40
from sklearn.model_selection import train_test_split
훈련용_data, 테스트용_data, 훈련용_target, 테스트용_target = train_test_split(data, target, test_size=0.25, random_state=40)
```

```python
# 데이터 구조(shape) 확인
print(훈련용_data.shape)
print(테스트용_data.shape)
```

```
(11, 2)
(4, 2)

```

```python
# Step 5. 분석하여 모델을 생성합니다.
from sklearn.neighbors import KNeighborsClassifier

# knn 모델 생성
knn = KNeighborsClassifier(n_neighbors=3) 
# k-nn : k 값을 필수로 지정해주어야 함.

# 모델 학습
knn.fit(훈련용_data, 훈련용_target)

# 모델 평가
knn.score(테스트용_data, 테스트용_target)
```

```python
# Step 6. 모델이 정확한지 임의의 데이터로 테스트합니다.
print( knn.predict([[1000, 15]]))
```

```
['수박']

```

```python
# Step 7. 위 데이터의 값을 그래프로 출력하여 확인합니다.
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic') 

plt.scatter(훈련용_data[:,0], 훈련용_data[:,1])
plt.scatter(1000, 15, marker='o')
plt.xlabel('무게')
plt.ylabel('길이')
plt.show()
```

```
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
C:\Users\user\AppData\Roaming\Python\Python313\site-packages\IPython\core\pylabtools.py:170: UserWarning: Glyph 47924 (\N{HANGUL SYLLABL
... (출력 생략됨) ...
```

![7._KNN_수박_참외_맞추기_공유용_img2.png](/assets/img/study/7._KNN_수박_참외_맞추기_공유용_img2.png)

```python
# Step 8. 최적의 k 값 찾기
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic') 

k_list = range(1,12)
accuracies = []

for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(훈련용_data, 훈련용_target.values.ravel())
  accuracies.append(classifier.score(테스트용_data, 테스트용_target))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("최적의 이웃 값 찾기")
plt.show()
```

```
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBarunGothic' not found.
findfont: Font family 'NanumBaru
... (출력 생략됨) ...
```

![7._KNN_수박_참외_맞추기_공유용_img3.png](/assets/img/study/7._KNN_수박_참외_맞추기_공유용_img3.png)

```python
# 최적의 K 값 가지고 실행

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(훈련용_data, 훈련용_target)
knn.score(테스트용_data, 테스트용_target)   
print(knn.predict([[1000, 15]]))
```

```
['수박']

```

```python
#학습된 모델 저장하기오 불러와서 재사용하기
import joblib


# 모델을 피클 파일로 저장합니다.
joblib.dump(knn, 'knn_model.pkl')
```

```python
import joblib

# 모델을 불러서 다시 사용하기
kn2 = joblib.load('knn_model.pkl')
print(kn2.predict([[800,8]]))
```

```
['참외']

```

