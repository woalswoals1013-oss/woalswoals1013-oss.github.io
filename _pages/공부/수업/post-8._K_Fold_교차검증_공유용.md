---
title: "8. K_Fold_교차검증"
date: 2026-06-18
---

# Google Colab 데이터 로드

```python
#Step 1. 구글 코랩에 한글 폰트 설정하기

!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

```python
#Step 2.분석할 데이터가 저장된 파일을 불러와서 변수에 할당합니다.
from google.colab import files
myfile = files.upload()
import io
import pandas as pd
#pd.read_csv로 csv파일 불러오기
src_data = pd.read_csv(io.BytesIO(myfile['의사결정나무_과일종류_2가지.csv']),
                       encoding='cp949')
src_data
```

# 로컬 데이터 로드

```python
#컴퓨터에서 작업하려면 아래 코드의 주석을 제거하고 실행하면 됩니다
import pandas as pd
src_data = pd.read_csv('./머신러닝실습용자료/의사결정나무_과일종류_2가지.csv',encoding='cp949')
src_data
```

# 공통 실습 코드

```python
#Step 3.주어진 데이터를 훈련용 데이터와 검증용 데이터로 나눕니다
import numpy as np
import pandas as pd

# 무게, 길이로 종류를 예측
# Dframe으로 읽어서. to numpy로 넘파이 배열로 변환
data = src_data[['무게', '길이']].to_numpy()
target = src_data[['종류']].to_numpy()
print(data)
print(target)

# train, test 데이터 분리
from sklearn.model_selection import train_test_split
훈련용_data, 테스트용_data, 훈련용_target, 테스트용_target = train_test_split(data, target, test_size=0.2, random_state=10)
```

```
[[2000.    30. ]
 [2500.    25. ]
 [1800.    20. ]
 [1500.    16. ]
 [1900.    19. ]
 [ 600.     9. ]
 [ 500.     8. ]
 [ 400.     7.5]
 [ 450.     5. ]
 [ 400.     4.5]
 [ 600.     9.5]
 [ 550.     8.5]]
[['수박']
 ['수박']
 ['수박']
 ['수박']
 ['수박']
 ['수박']
 ['참외']
 ['참외']
 ['참외']
 ['참외']
 ['참외']
 ['참외']]

```

```python
# 교차검증 없이 모델 검증합니다.
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 의사결정트리 분류기 모델 생성
dt = DecisionTreeClassifier(random_state=10)

# 모델 학습
dt.fit(훈련용_data, 훈련용_target)

# 훈련용 데이터 기준 score 출력
print('훈련용 데이터 기준 정확도 :', dt.score(훈련용_data, 훈련용_target))

# 테스트용 데이터 기준 score 출력
print('테스트용 데이터 기준 정확도 :', dt.score(테스트용_data, 테스트용_target))

```

```
훈련용 데이터 기준 정확도 : 1.0
테스트용 데이터 기준 정확도 : 1.0

```

```python
#3-Fold 교차 검증 수행
from sklearn.model_selection import cross_validate , cross_val_score
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=10)

# cross_validate() : 검사 결과를 자세하게 보여주는 것
scores_1 = cross_validate(dt, 훈련용_data, 훈련용_target, cv=3)
# cross_val_score() : 검사 결과 중에서 평균 점수만 보여주는 것
scores_2 = cross_val_score(dt, 훈련용_data, 훈련용_target, cv=3)

print('cross_validate 결과:', scores_1)
print('cross_validate 결과:', np.mean(scores_1['test_score']))
print('cross_val_score 결과:', np.mean(scores_2) )
```

```
cross_validate 결과: {'fit_time': array([0.0011394 , 0.00114346, 0.00066853]), 'score_time': array([0.00044489, 0.00053668, 0.00041699]), 'test_score': array([0.66666667, 1.        , 0.66666667])}
cross_validate 결과: 0.7777777777777777
cross_val_score 결과: 0.7777777777777777

```

```python
#5-Fold 교차검증 수행
from sklearn.model_selection import cross_validate , cross_val_score
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=10)

# cross_validate() : 검사 결과를 자세하게 보여주는 것
scores_1 = cross_validate(dt, 훈련용_data, 훈련용_target, cv=5)
# cross_val_score() : 검사 결과 중에서 평균 점수만 보여주는 것
scores_2 = cross_val_score(dt, 훈련용_data, 훈련용_target, cv=5)

print('cross_validate 결과:', scores_1)
print('cross_validate 결과:', np.mean(scores_1['test_score']))
print('cross_val_score 결과:', np.mean(scores_2) )


```

```
cross_validate 결과: {'fit_time': array([0.00177217, 0.00061226, 0.00055718, 0.00050354, 0.00073457]), 'score_time': array([0.00052023, 0.00038433, 0.0003643 , 0.00041318, 0.00041556]), 'test_score': array([0.5, 1. , 0.5, 1. , 1. ])}
cross_validate 결과: 0.8
cross_val_score 결과: 0.8

```

```
c:\Users\user\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\model_selection\_split.py:813: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
  warnings.warn(
c:\Users\user\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\model_selection\_split.py:813: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
  warnings.warn(

```

