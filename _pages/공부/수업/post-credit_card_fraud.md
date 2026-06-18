---
title: "실전 실습_ 신용카드 이상 탐지 (Feature Engineering)"
permalink: /study/class/credit_card_fraud/
date: 2026-06-18
---

# 실습 1 신용카드 이상 탐지

데이터 준비
실습에 사용될 데이터는 Kaggle의 Credit Card Fraud Detection 데이터셋입니다. 이 데이터셋은 거래의 시간, 금액과 함께 28개의 PCA 변환된 특성들을 포함하고 있습니다. 'Class' 레이블은 사기 거래를 나타내는 1과 정상 거래를 나타내는 0으로 구분됩니다.

데이터를 불러오고, 전처리하는 기본적인 코드는 아래와 같습니다:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 데이터를 불러옵니다.
data = pd.read_csv('creditcard.csv')
data
```

## 추가: 랜덤 포레스트 특성 중요도(Feature Importance) 분석

```python
import pandas as pd

# 랜덤 포레스트가 학습하면서 판단한 각 변수의 중요도 추출
importances = rf.feature_importances_

# 보기 쉽게 그래프로 그리기
feat_importances = pd.Series(importances, index=X.columns)
feat_importances.sort_values(ascending=True).plot(kind='barh', title='Feature Importances (Random Forest)')
plt.show()

# [참고] 이 막대 그래프에서 막대기가 가장 짧은(중요도가 0에 가까운) 변수들은
# 예측에 쓸모가 없으므로 다음 학습 때 X에서 drop 시키면 속도와 성능을 모두 개선할 수 있습니다.
```

```python
# [참고: 모든 분류 문제의 공통 전처리 시작점]
# y 데이터(class)의 분포를 확인합니다.
print("Class Distribution:")
print(data['Class'].value_counts())

# 교수님의 지시대로 V1부터 V10까지만 feature로 사용합니다.
X = data.loc[:, 'V1':'V10']
y = data['Class']
X.head()
```

## 추가: 상관관계(Correlation) 히트맵 분석

```python
import seaborn as sns
import matplotlib.pyplot as plt

# V1~V10 그리고 Class 변수 간의 상관관계 행렬 계산
corr_matrix = X.join(y).corr()

# 보기 쉽게 히트맵(Heatmap)으로 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (V1~V10 & Class)")
plt.show()

# [참고] 만약 이 그래프에서 0.8 혹은 -0.8 이상/이하의 높은 수치가 나온다면,
# 두 특성이 겹친다는 뜻이므로 하나를 삭제(drop)하거나 평균내어 병합할 수 있습니다.
```

```python
# [참고: 1. 선형회귀분석_공유용.ipynb 등 회귀/분류 전반에서 쓰인 train_test_split 공통 패턴]
# 데이터를 훈련 세트와 테스트 세트로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# [참고: 5. 로지스틱회귀분석_공유용.ipynb 에서 피처의 단위를 맞추기 위해 사용했던 정규화 기법의 연장선]
# 데이터 표준화 작업을 실시합니다,
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
from sklearn.metrics import accuracy_score, precision_score, f1_score

# [참고: 5. 로지스틱회귀분석_공유용.ipynb의 LogisticRegression() 학습 및 평가 로직 차용]
# 로지스틱 회귀 모델을 생성하고 학습합니다.
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

# 학습된 모델로 테스트 데이터를 예측하고 평가합니다.
예측결과 = lr.predict(X_test_scaled)

# [참고: 분류 모델의 성능을 평가하기 위한 지표들]
# 정확도, 정밀도, F1 Score를 계산 및 출력합니다.
print("정확도(Accuracy):", accuracy_score(y_test, 예측결과))
print("정밀도(Precision):", precision_score(y_test, 예측결과))
print("F1 점수:", f1_score(y_test, 예측결과))

# AUC 점수를 계산합니다. (템플릿 원본 코드)
lr_auc_score = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
print("Logistic Regression AUC score:", lr_auc_score)
```

## Decision Tree 로 직접해보기

```python
# [참고: 기존 트리/비지도학습 파트의 기본 개념(DecisionTree)을 활용한 모델 구성]
# 결정 트리 모델을 생성하고 학습합니다.
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# [참고: 앞선 Logistic Regression과 동일한 평가 패턴 적용]
# 학습된 모델로 테스트 데이터를 예측하고 평가합니다.
dt_예측결과 = dt.predict(X_test_scaled)

# AUC 점수를 계산합니다.
dt_auc_score = roc_auc_score(y_test, dt.predict_proba(X_test_scaled)[:, 1])
print("Decision Tree AUC score:", dt_auc_score)
```

## Random Forest 로 해보기

```python
# [참고: '재민/트리, 비지도학습/2. 앙상블_랜덤포레스트_과일종류맞추기_공유용.ipynb' 의 앙상블 학습 로직 차용]
# 랜덤 포레스트 모델을 생성하고 학습합니다.
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# 학습된 모델로 테스트 데이터를 예측하고 평가합니다.
rf_예측결과 = rf.predict(X_test_scaled)

# AUC 점수를 계산합니다.
rf_auc_score = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
print("Random Forest AUC score:", rf_auc_score)
```

## 퀴즈) SVM 사용해보기

```python
from sklearn.svm import SVC

# [주의] 신용카드 데이터셋은 약 28만 개로 매우 커서 SVM을 전체 데이터에 돌리면 몇 시간이 걸릴 수 있습니다.
# 따라서 실습을 위해 훈련 데이터의 5,000개만 샘플링하여 빠르게 학습시킵니다.
# 모델 생성 및 학습 (확률 출력을 위해 probability=True 설정)
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled[:5000], y_train[:5000])

# AUC 점수 계산
svm_auc_score = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1])
print("SVM AUC score (Sampled):", svm_auc_score)
```

# 가장 AUC점수가 높았던 모델 GridSearch로 튜닝하기

```python
from sklearn.model_selection import GridSearchCV

# [참고: '재민/회귀, 분류/9. 그리드서치_공유용.ipynb' 의 하이퍼파라미터 최적화(GridSearchCV) 로직 차용]
# 튜닝할 하이퍼파라미터 설정
params = {
    'max_depth': [3, 5, 7]
}

# GridSearchCV 실행 (시간 절약을 위해 n_jobs=-1 사용)
gs = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train_scaled, y_train)

# [참고: 9. 그리드서치_공유용.ipynb 에서 사용했던 best_params_ 속성을 활용한 최적의 결과 도출]
print("최적의 파라미터:", gs.best_params_)
print("최고 AUC 점수:", gs.best_score_)
```

