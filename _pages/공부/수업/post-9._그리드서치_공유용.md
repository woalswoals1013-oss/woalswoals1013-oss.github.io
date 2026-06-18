---
title: "9. 그리드서치"
permalink: /study/class/9._그리드서치_공유용/
date: 2026-06-18
---

## GridSearch
최적의 하이퍼파라미터를 찾는 방법

```python
#컴퓨터에서 작업하려면 아래 코드의 주석을 제거하고 실행하면 됩니다
import pandas as pd
src_data = pd.read_csv('./머신러닝실습용자료/의사결정나무_과일종류_2가지.csv',encoding='cp949')
src_data
```

```python
#Step 3.주어진 데이터를 훈련용 데이터와 검증용 데이터로 나눕니다
import numpy as np

# 무게, 길이로 종류를 예측
data = src_data[['무게', '길이']]
target = src_data['종류']
print(data)
print(target)

# train, test 데이터 분리
from sklearn.model_selection import train_test_split
훈련용_data, 테스트용_data, 훈련용_target, 테스트용_target = train_test_split(data, target)
print(훈련용_data)
print(훈련용_target)
print(테스트용_data)
print(테스트용_target)

```

```
      무게    길이
0   2000  30.0
1   2500  25.0
2   1800  20.0
3   1500  16.0
4   1900  19.0
5    600   9.0
6    500   8.0
7    400   7.5
8    450   5.0
9    400   4.5
10   600   9.5
11   550   8.5
0     수박
1     수박
2     수박
3     수박
4     수박
5     수박
6     참외
7     참외
8     참외
9     참외
10    참외
11    참외
Name: 종류, dtype: str
      무게    길이
11   550   8.5
10   600   9.5
5    600   9.0
7    400   7.5
6    500   8.0
0   2000  30.0
3   1500  16.0
8    450   5.0
4   1900  19.0
11    참외
10    참외
5     수박
... (출력 생략됨) ...
```

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


parm = {'max_depth':[1,2,3]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=50) , parm , n_jobs=-1)
gs.fit(훈련용_data , 훈련용_target)
```

```
c:\Users\user\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\model_selection\_split.py:813: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
  warnings.warn(

```

```python
print(gs.best_params_)
dt = gs.best_estimator_
print(dt.score(훈련용_data , 훈련용_target))
```

```
{'max_depth': 1}
0.8888888888888888

```

```python
#한꺼번에 여러 속성값을 찾을 경우
from sklearn.model_selection import GridSearchCV
parm = {'max_depth': range(1,10,1) ,
        'min_impurity_decrease': np.arange(0.0001,0.001 , 0.0001),
        'min_samples_split' : range(2,100,10) }
gs = GridSearchCV(DecisionTreeClassifier(random_state=50) , parm , n_jobs=-1)
gs.fit(훈련용_data , 훈련용_target)
print(gs.best_params_)
```

```
c:\Users\user\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\model_selection\_split.py:813: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
  warnings.warn(

```

```
{'max_depth': 1, 'min_impurity_decrease': np.float64(0.0001), 'min_samples_split': 2}

```

```python
#교차검증 점수중 최고값을 확인하기
print(np.max(gs.cv_results_['mean_test_score']))
```

```
0.8

```

```python
# 최적의 모델로 테스트용 데이터로 최종 테스트하기
dt = gs.best_estimator_
print(dt.score(테스트용_data , 테스트용_target))
print(dt.score(훈련용_data , 훈련용_target))
```

```
1.0
0.8888888888888888

```

