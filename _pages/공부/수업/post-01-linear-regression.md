---
title: "1. 선형회귀분석 기초 (Linear Regression)"
date: 2026-06-29
thumbnail: "/assets/img/study/linear_reg_1.png"
---

이번 시간에는 머신러닝의 가장 기초가 되는 **선형회귀분석(Linear Regression)**에 대해 알아보고, 파이썬을 이용해 직접 데이터를 분석해 보겠습니다. 

우리가 풀고자 하는 문제는 바로 **"공부 시간과 시험 점수 사이의 관계"**를 파악하는 것입니다.

---

## 1. 데이터 불러오기 및 확인

먼저, 분석할 데이터를 불러와야 합니다. 여기서는 Pandas 라이브러리를 이용해 CSV 파일을 읽어옵니다.

```python
import pandas as pd

# 로컬 PC에서 데이터 불러오기
study = pd.read_csv('./머신러닝실습용자료/공부시간과시험점수.csv', encoding='cp949')
study.head() # 데이터 상위 5개 확인
```

| 이름 | 공부시간 | 시험점수 |
|---|---|---|
| 이원재 | 15.0 | 85.0 |
| 맹승주 | 14.5 | 86.5 |
| 안미경 | 14.0 | 86.0 |
| ... | ... | ... |

* **이름**: 분석에 사용하지 않을 의미 없는 데이터입니다.
* **공부시간 (Feature)**: 우리가 결과를 예측하기 위해 사용할 특징 데이터입니다.
* **시험점수 (Target)**: 우리가 맞추고자 하는 결과값(정답) 데이터입니다.

---

## 2. 데이터 분리하기 (Train / Test)

머신러닝 모델을 만들 때는 모델이 잘 학습했는지 평가하기 위해 데이터를 **훈련용(Train)**과 **테스트용(Test)**으로 나누어야 합니다.

```python
from sklearn.model_selection import train_test_split
import numpy as np

# pandas 데이터를 numpy 배열로 변환
data = study['공부시간'].to_numpy()
target = study['시험점수'].to_numpy()

# 훈련 세트와 테스트 세트로 8:2 비율로 나눕니다.
훈련용_data, 테스트용_data, 훈련용_target, 테스트용_target = train_test_split(
    data, target, test_size=0.2, random_state=40
)
```

---

## 3. 데이터 형태 맞추기 (Reshape)

우리가 방금 나눈 공부 시간 데이터는 단순한 1차원 배열(리스트 형태)입니다. 하지만 사이킷런(Scikit-learn)의 머신러닝 모델들은 **2차원 배열** 형태의 데이터를 요구합니다. 

따라서 `reshape` 함수를 이용해 데이터의 모양을 바꿔주어야 합니다.

```python
# reshape(-1, 1)은 열(Column)을 1개로 고정하고, 
# 남은 데이터 개수만큼 행(Row)을 알아서 맞추라는 뜻입니다.
훈련용_data = 훈련용_data.reshape(-1, 1)
테스트용_data = 테스트용_data.reshape(-1, 1)

print(훈련용_data.shape) # 출력 결과: (20, 1)
```

---

## 4. 선형회귀 모델 학습 및 평가

이제 데이터를 준비했으니, 실제 선형회귀 모델을 불러와서 학습(`fit`)을 진행해 봅시다!

```python
from sklearn.linear_model import LinearRegression

# 모델 객체 생성
lr = LinearRegression()

# 훈련용 데이터를 이용해 모델 학습
lr.fit(훈련용_data, 훈련용_target)
```

학습이 끝났다면 모델이 얼마나 잘 예측하는지 점수(`score`)를 확인해볼까요? 이 점수는 결정계수($R^2$)를 의미하며, 1에 가까울수록 모델이 정확하다는 뜻입니다.

```python
# 모델 정확도 평가
print("훈련 세트 점수:", lr.score(훈련용_data, 훈련용_target))
print("테스트 세트 점수:", lr.score(테스트용_data, 테스트용_target))

# 예측해보기: 만약 16시간을 공부했다면 몇 점을 받을까?
print("16시간 공부 시 예상 점수:", lr.predict([[16]]))
```

> **예상 결과** <br> 16시간 공부 시 예상 점수: 약 90.1점

---

## 5. 결과 시각화

모델이 찾아낸 "공부 시간과 시험 점수의 관계(기울기와 절편)"를 맷플롯립(Matplotlib) 그래프로 예쁘게 그려보겠습니다.

```python
import matplotlib.pyplot as plt

# 원본 훈련 데이터 산점도
plt.scatter(훈련용_data, 훈련용_target)

# 모델이 학습한 일차방정식 직선 (회귀선) 그리기
# x를 5부터 18까지 그릴 때, y는 모델의 기울기(coef_)와 절편(intercept_)을 따름
plt.plot([5, 18], [5*lr.coef_ + lr.intercept_, 18*lr.coef_ + lr.intercept_], color='red')

# 16시간 공부했을 때의 예측값 위치 (세모 모양)
plt.scatter(16, 90.1, marker="^", color='green', s=100)

plt.show()
```

![output](/assets/img/study/linear_reg_1.png)

위 빨간 선이 바로 인공지능이 찾아낸 **가장 최적의 회귀선**입니다!

---

## 6. (심화) 다항 회귀 분석

만약 데이터가 직선이 아니라 약간의 곡선 형태를 띤다면 어떨까요? 이럴 때는 단순히 1차 방정식이 아니라, 데이터의 제곱 값을 추가하여 **다항 회귀분석(Polynomial Regression)**을 적용할 수 있습니다.

```python
# 훈련 데이터의 제곱값을 기존 데이터 앞에 붙여줍니다.
훈련용_data_poly = np.column_stack((훈련용_data ** 2, 훈련용_data))
테스트용_data_poly = np.column_stack((테스트용_data ** 2, 테스트용_data))

# 다시 학습
lr = LinearRegression()
lr.fit(훈련용_data_poly, 훈련용_target)

# 점수 재확인
print("다항 회귀 훈련 세트 점수:", lr.score(훈련용_data_poly, 훈련용_target))
print("다항 회귀 테스트 세트 점수:", lr.score(테스트용_data_poly, 테스트용_target))
```

이렇게 하면 모델이 조금 더 복잡한 곡선의 패턴까지도 찾아낼 수 있게 됩니다.
