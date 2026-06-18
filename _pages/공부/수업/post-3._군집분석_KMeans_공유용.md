---
title: "3. 군집분석_KMeans"
date: 2026-06-26
thumbnail: "/assets/img/study/3._군집분석_KMeans_공유용_img1.png"
---

# Google Colab 데이터 로드

```python
#Step 1.분석할 데이터가 저장된 파일을 불러와서 변수에 할당합니다.
from google.colab import files
myfile = files.upload()
import io
import pandas as pd
#pd.read_csv로 csv파일 불러오기
fruits = pd.read_csv(io.BytesIO(myfile['과일3개.csv']),
                       encoding='cp949')
fruits
```

# 로컬 데이터 로드

```python
#컴퓨터에서 작업하려면 아래 코드의 주석을 제거하고 실행하면 됩니다
import pandas as pd
fruits = pd.read_csv('./머신러닝실습용자료/과일3개.csv',encoding='cp949')
fruits
```

# 공통 실습 코드

```python
# Step 2.데이터의 분포를 그림으로 그리고 임의의 중심점 지정
import matplotlib.pyplot as plt
x1,y1 = 2000, 22
x2,y2 = 200, 2.5
x3,y3 = 500, 10

data = fruits[['무게_g','길이_cm']]
plt.figure(figsize=(7,5))
plt.title("Before", fontsize=15)
plt.plot(data["무게_g"], data["길이_cm"], "o", label="Data")
plt.plot([x1,x2,x3], [y1,y2,y3], "rD", \
         marker='*', markersize=12, label='init_Centroid')
plt.xlabel("Weight", fontsize=12)
plt.ylabel("Length", fontsize=12)
plt.legend()
plt.grid()
plt.show()
```

```
C:\Users\user\AppData\Local\Temp\ipykernel_33832\4249871694.py:11: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "rD" (-> marker='D'). The keyword argument will take precedence.
  plt.plot([x1,x2,x3], [y1,y2,y3], "rD", \

```

![3._군집분석_KMeans_공유용_img1.png](/assets/img/study/3._군집분석_KMeans_공유용_img1.png)

```python
# Step 3. 군집 분석을 수행합니다.
from sklearn.cluster import KMeans
import numpy as np

# 군집 분석은 비지도학습이기에, target이 없습니다.
data = fruits[['무게_g','길이_cm']]

#초기의 점을 지정할 경우
kmeans = KMeans(n_clusters=3, init = np.array([(x1,y1),(x2,y2),(x3,y3)]))

#초기의 점을 지정하지 않을 경우
#kmeans = KMeans(n_clusters=3)

# 모델 학습
kmeans.fit(data)

# k-means의 라벨과, 중심점 좌표 가져오기
data['cluster'] = kmeans.labels_
final_centroid = kmeans.cluster_centers_
```

```python
data
```

```python
#Step 4. 군집화를 진행하여 최종 결과를 확인합니다.
plt.figure(figsize=(7,5))
plt.title("After", fontsize=15)
plt.scatter(data['무게_g'],data['길이_cm'],c=data['cluster'])
plt.plot(final_centroid[:,0], final_centroid[:,1], "rD", \
         marker='*',markersize=12, label='final_Centroid')
plt.xlabel("Weight", fontsize=12)
plt.ylabel("Length", fontsize=12)
plt.legend()
plt.grid()
plt.show()
```

```
C:\Users\user\AppData\Local\Temp\ipykernel_33832\379812364.py:5: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "rD" (-> marker='D'). The keyword argument will take precedence.
  plt.plot(final_centroid[:,0], final_centroid[:,1], "rD", \

```

![3._군집분석_KMeans_공유용_img2.png](/assets/img/study/3._군집분석_KMeans_공유용_img2.png)

```python
# 각각의 요소에 대한 라벨 출력
print(kmeans.labels_)
```

```
[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2]

```

```python
# [500, 20] 데이터 넣었을 때 예측값 확인
kmeans.predict([[500, 20]])
```

```
c:\Users\user\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\validation.py:2691: UserWarning: X does not have valid feature names, but KMeans was fitted with feature names
  warnings.warn(

```

```python
# [1700, 15] 데이터 넣었을 때 예측값 확인
kmeans.predict([[1700, 15]])
```

```
c:\Users\user\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\validation.py:2691: UserWarning: X does not have valid feature names, but KMeans was fitted with feature names
  warnings.warn(

```

```python
# 클러스터 중심과 클러스터에 속한 샘플 사이의 거리의 제곱 합 출력
print(kmeans.inertia_)
```

```
610236.1279999999

```

```python
#최적의 군집 개수 찾기 - Elbow Method
import matplotlib.pyplot as plt

inertia = [ ]
for i in range(2,15) :
  km = KMeans(n_clusters=i)
  km.fit(data)
  inertia.append(km.inertia_)
plt.plot(range(2,15) , inertia)
plt.show()
```

![3._군집분석_KMeans_공유용_img3.png](/assets/img/study/3._군집분석_KMeans_공유용_img3.png)

