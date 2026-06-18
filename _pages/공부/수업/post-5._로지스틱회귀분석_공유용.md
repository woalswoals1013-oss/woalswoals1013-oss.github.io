---
title: "5. 로지스틱회귀분석"
permalink: /study/class/5._로지스틱회귀분석_공유용/
date: 2026-06-18
---

# Google Colab 데이터 로드

```python
#Step 1.분석할 데이터가 저장된 파일을 불러와서 변수에 할당합니다.
from google.colab import files
myfile = files.upload()
import io
import pandas as pd
#pd.read_csv로 csv파일 불러오기
study = pd.read_csv(io.BytesIO(myfile['공부시간과시험합격.csv']),
                       encoding='cp949')
study
```

# 로컬 데이터 로드

```python
#컴퓨터에서 작업하려면 아래 코드의 주석을 제거하고 실행하면 됩니다
import pandas as pd
study = pd.read_csv('./머신러닝실습용자료/공부시간과시험합격.csv',encoding='cp949')
study
```

# 공통 실습 코드

```python
#Step 2: 훈련용 데이터셋과 테스트용 데이터셋 나누어서 분석
from sklearn.model_selection import train_test_split

# data  나누기 공부시간을 가지고 합격여부를 예측
data = study['공부시간']
target = study['합격여부']

# train, test 데이터 분리
훈련용_data, 테스트용_data, 훈련용_target, 테스트용_target = train_test_split(data, target, test_size=0.2, random_state=40)
```

```python
#Step 3. 학습 후 모델을 생성하고 예측을 수행합니다
from sklearn.linear_model import LogisticRegression
# 로지스틱회귀 모델 생성
lr = LogisticRegression()

# 행(row)으로 되어있는 데이터, 열(column)로 나열
훈련용_data = 훈련용_data.reshape(-1, 1)
테스트용_data = 테스트용_data.reshape(-1, 1)

# 모델 학습
lr.fit(훈련용_data, 훈련용_target)

# 테스트용_data로 예측
print(테스트용_data)
print(lr.predict(테스트용_data))
```

```
[[9. ]
 [7. ]
 [9.3]
 [8.4]
 [9.5]]
['불합격' '불합격' '불합격' '불합격' '합격']

```

```python
import numpy as np

# 각 항목별 확률값 출력
print(np.round(lr.predict_proba(테스트용_data),3))
print(lr.predict_proba(테스트용_data))  
```

```
[[0.668 0.332]
 [0.984 0.016]
 [0.546 0.454]
 [0.849 0.151]
 [0.461 0.539]]
[[0.66809273 0.33190727]
 [0.98418294 0.01581706]
 [0.54609343 0.45390657]
 [0.8492738  0.1507262 ]
 [0.4605282  0.5394718 ]]

```

# Google Colab 데이터 로드

```python
# 다중 분류 활용-과일 종류 분류하기
#Step 1.분석할 데이터가 저장된 파일을 불러와서 변수에 할당합니다.
from google.colab import files
myfile = files.upload()
import io
import pandas as pd
#pd.read_csv로 csv파일 불러오기
fruit_2 = pd.read_csv(io.BytesIO(myfile['과일채소목록_2.csv']),
                       encoding='cp949')
fruit_2
```

# 로컬 데이터 로드

```python
#컴퓨터에서 작업하려면 아래 코드의 주석을 제거하고 실행하면 됩니다
import pandas as pd
fruit_2 = pd.read_csv('./머신러닝실습용자료/과일채소목록_2.csv',encoding='cp949')
fruit_2
```

# 공통 실습 코드

```python
#Step 2: 훈련용 데이터셋과 테스트용 데이터셋 나누어서 분석
from sklearn.model_selection import train_test_split

# data  나누기 무게, 길이, 당도를 가지고 과일 종류 분류
data = fruit_2[['무게_g' ,'길이_cm' ,'당도']]
target = fruit_2['종류']

# train, test 데이터 분리
훈련용_data, 테스트용_data, 훈련용_target, 테스트용_target = train_test_split(data, target, test_size=0.2, random_state=40)
```

```python
#Step 3. 데이터 표준화를 진행합니다
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#훈련용 데이터에 대해 평균은 0 표준편차는 1을 갖도록 데이터를 정리
ss.fit(훈련용_data)

#data에 대해서 표준화 적용
표준화_훈련용_data = ss.transform(훈련용_data)
표준화_테스트용_data = ss.transform(테스트용_data)
```

```python
# 모델을 생성하고 테스트하고 성능을 확인합니다.
from sklearn.linear_model import LogisticRegression
import numpy as np

# 로지스틱회귀분석 모델 생성 및 학습
softmax_reg = LogisticRegression()
softmax_reg.fit(표준화_훈련용_data, 훈련용_target)


# 분류 결과 확인
print(softmax_reg.predict(표준화_테스트용_data))

# 분류 확률 확인
print(np.round(softmax_reg.predict_proba(표준화_테스트용_data),3))

# 분류 점수 확인
print(softmax_reg.score(표준화_테스트용_data, 테스트용_target))
                        
```

```
['자두' '옥수수' '참외' '자두' '참외' '거봉포도' '수박' '거봉포도' '수박' '거봉포도']
[[0.021 0.007 0.018 0.658 0.297]
 [0.021 0.008 0.945 0.009 0.017]
 [0.039 0.03  0.052 0.421 0.458]
 [0.021 0.006 0.007 0.671 0.296]
 [0.067 0.046 0.032 0.334 0.521]
 [0.93  0.031 0.011 0.003 0.026]
 [0.053 0.667 0.012 0.006 0.263]
 [0.877 0.04  0.021 0.009 0.052]
 [0.004 0.979 0.004 0.    0.013]
 [0.893 0.037 0.029 0.006 0.035]]
1.0

```

