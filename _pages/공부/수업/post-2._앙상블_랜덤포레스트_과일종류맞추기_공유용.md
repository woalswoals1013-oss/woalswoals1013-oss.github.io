---
title: "2. 앙상블_랜덤포레스트_과일종류맞추기"
date: 2026-06-18
thumbnail: "/assets/img/study/2._앙상블_랜덤포레스트_과일종류맞추기_공유용_img1.png"
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
과일채소목록 = pd.read_csv(io.BytesIO(myfile['과일채소목록.csv']),
                       encoding='cp949')
과일채소목록
```

# 로컬 데이터 로드

```python
#컴퓨터에서 작업하려면 아래 코드의 주석을 제거하고 실행하면 됩니다
import pandas as pd
src_data = pd.read_csv('./머신러닝실습용자료/과일채소목록.csv',encoding='cp949')
src_data
```

# 공통 실습 코드

```python
#Step 3. 훈련용 세트와 테스트용 세트로 나눕니다.
# '무게_g','길이_cm','색상','당도'에 따른 과일종류 분류
data = src_data[['무게_g','길이_cm','색상','당도']]
target = src_data['종류']

# train, test 데이터 분리
from sklearn.model_selection import train_test_split
훈련용_data, 테스트용_data, 훈련용_target, 테스트용_target = train_test_split(data, target, test_size=0.3, random_state=40)

```

```python
# 각각의 데이터 확인
print(훈련용_data.shape , 테스트용_data.shape)
print(훈련용_data)
print(훈련용_target)
```

```
(35, 4) (15, 4)
    무게_g  길이_cm  색상   당도
41   501   25.1   1  2.1
23   270   26.0   3  8.5
36   401    7.6   2  7.3
5    100    3.5   3  6.0
13   400    6.5   2  6.5
39   601    8.6   2  8.1
17   380   22.0   1  1.5
43   401   23.1   1  1.1
24   290   29.0   3  9.0
3   1500   16.0   1  8.5
22   220   22.0   3  7.0
40   451   20.1   1  3.1
26  2501   25.1   1  7.1
34   111    3.7   3  7.6
20   280   28.0   3  8.0
28  1501   16.1   1  8.6
14   600    8.5   2  8.0
15   450   20.0   1  3.0
30   101 
... (출력 생략됨) ...
```

```python
from sklearn.ensemble import RandomForestClassifier
# 랜덤 포레스트 모델 생성
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=40) #n_estimators : 트리의 개수, n_jobs : 사용할 CPU 코어 수, random_state : 랜덤 시드 설정

# 학습
rf.fit(훈련용_data, 훈련용_target)

# 예측
print(rf.predict(테스트용_data))

# score
print(rf.score(테스트용_data, 테스트용_target))
```

```
['자두' '수박' '거봉포도' '참외' '거봉포도' '수박' '옥수수' '수박' '참외' '수박' '옥수수' '참외' '수박'
 '거봉포도' '옥수수']
1.0

```

### 결과표 작성 및 시각화

```python
# 테스트 데이터 확인
테스트용_data
```

```python
from sklearn.metrics import classification_report

#예측 
pred = rf.predict(테스트용_data)

#리포트 출력 
print(classification_report(테스트용_target, pred))
```

```
              precision    recall  f1-score   support

        거봉포도       1.00      1.00      1.00         3
          수박       1.00      1.00      1.00         5
         옥수수       1.00      1.00      1.00         3
          자두       1.00      1.00      1.00         1
          참외       1.00      1.00      1.00         3

    accuracy                           1.00        15
   macro avg       1.00      1.00      1.00        15
weighted avg       1.00      1.00      1.00        15


```

```python
# 예측결과 데이터프레임을 만들고
예측결과 = pd.DataFrame(rf.predict(테스트용_data), columns=['예측결과'])

# concat을 통해 기존 테스트 data와 예측결과 데이터를 합친다.
result = pd.concat([테스트용_data.reset_index(drop=True), 예측결과], axis=1)
result
```

```python
# k-fold 교차 검증
from sklearn.model_selection import cross_validate


# cross_validate() : 검사 결과를 자세하게 보여주는 것
scores = cross_validate(rf, data, target, cv=5, return_train_score=True)
print(scores)




```

```
{'fit_time': array([0.18425727, 0.15477157, 0.15879893, 0.13109159, 0.12138104]), 'score_time': array([0.03266931, 0.02145076, 0.03545213, 0.03197932, 0.02150893]), 'test_score': array([1., 1., 1., 1., 1.]), 'train_score': array([1., 1., 1., 1., 1.])}

```

```python
# 중요 속성 지표값 출력

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
font_location = "C:/Windows/Fonts/malgun.ttf"
#plt.rc('font', family='NanumBarunGothic')


# 혹시 위 폰트가 에러 날 경우 폰트 사용하면 됩니다
font_name = fm.FontProperties(fname = font_location).get_name()
matplotlib.rc('font', family=font_name)

imp = rf.feature_importances_
print('중요속성지표값:',imp)

plt.figure()
plt.bar(range(len(imp)),imp)
plt.xticks(range(len(imp)),data.columns, rotation=90)
plt.show()
```

```
중요속성지표값: [0.31303771 0.29685569 0.22213184 0.16797476]

```

![2._앙상블_랜덤포레스트_과일종류맞추기_공유용_img1.png](/assets/img/study/2._앙상블_랜덤포레스트_과일종류맞추기_공유용_img1.png)

