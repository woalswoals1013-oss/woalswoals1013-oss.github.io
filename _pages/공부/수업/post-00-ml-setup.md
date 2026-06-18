---
title: "0. 머신러닝 기본 환경 세팅"
permalink: /study/class/00-ml-setup/
date: 2026-06-18
---

# 머신러닝 기초 세팅 가이드

이 포스트는 앞으로 진행될 모든 머신러닝 실습(회귀, 분류, 군집화 등)에서 공통적으로 들어가는 **구글 코랩 환경 설정 및 한글 폰트 적용** 코드를 정리한 글입니다.

매 실습 파일마다 길게 들어가 있는 환경 설정 코드는 이 글을 참고하시면 됩니다.

## 1. 구글 코랩 한글 폰트 설정
맷플롯립(Matplotlib) 등에서 그래프를 그릴 때 한글이 깨지지 않도록 나눔고딕 폰트를 설치합니다.
```python
# 구글 코랩에 한글 폰트 설정하기
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```
(이후 런타임 다시 시작 필요)

## 2. 공통 라이브러리 및 폰트 적용
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 적용 (런타임 재시작 후 실행)
plt.rc('font', family='NanumBarunGothic') 

# 마이너스 기호 깨짐 방지
plt.rc('axes', unicode_minus=False)
```

## 3. 데이터 분할 공통 패턴
```python
from sklearn.model_selection import train_test_split

# 모델이 학습할 특징(X)과 맞춰야 할 정답(y) 분리
X = data.drop('target', axis=1)
y = data['target']

# 훈련용 80%, 테스트용 20% 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
