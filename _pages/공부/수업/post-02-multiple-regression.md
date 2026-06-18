---
title: "2. 다중회귀분석_공부시간시험성적"
date: 2025-06-28
thumbnail: "/assets/img/thumbnail/default_ml.png"
---

# Google Colab 데이터 로드

```python
#Step 1.분석할 데이터가 저장된 파일을 불러와서 변수에 할당합니다.
from google.colab import files
myfile = files.upload()
import io
import pandas as pd
#pd.read_csv로 csv파일 불러오기
study = pd.read_csv(io.BytesIO(myfile['공부시간과시험점수2.csv']),
                       encoding='cp949')
study
```

# 로컬 데이터 로드

```python
#컴퓨터에서 작업하려면 아래 코드의 주석을 제거하고 실행하면 됩니다
import pandas as pd
study = pd.read_csv('./머신러닝실습용자료/공부시간과시험점수2.csv',encoding='cp949')
study
```

# 공통 실습 코드

```python
#Step 2: 훈련용 데이터셋과 테스트용 데이터셋 나누어서 분석
from sklearn.model_selection import train_test_split

# data, target 지정
# 다중회귀분석의 경우 data를 여러개 설정.
data = study[['공부시간', '학원수','과외여부']]
target = study['시험점수']

# train, test 데이터 분리
훈련용_data, 테스트용_data, 훈련용_target, 테스트용_target = train_test_split(data, target, test_size=0.2, random_state=40)

```

```python
# 선형회귀분석 학습
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(훈련용_data, 훈련용_target)




# 13, 5, 0이라는 값을 넣어 예측
lr.predict([[13,5,0]])
```

```
c:\Users\user\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\validation.py:2691: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
  warnings.warn(

```

```python
# 테스트 데이터로 스코어 확인
lr.score(테스트용_data , 테스트용_target)
```

