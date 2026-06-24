---
title: "실전 실습_ 신용카드 이상 탐지 (초고속 모델 최적화 파이프라인)"
date: 2026-06-25
thumbnail: "/assets/img/thumbnail/default_ml.png"
---

# 실습 2 신용카드 이상 탐지 초고속 파이프라인

본 실습에서는 이전 신용카드 이상 탐지 실습을 고도화하여, 시험 제약 조건인 **5개 피처 자동 탐색** 및 **F1-Score 극대화**를 수행합니다. 
동시에 대용량 데이터셋(28만 행)에서도 **3분 이내에 학습 및 최적화가 완료**되도록 스크리닝 및 학습 프로세스를 최적화하였습니다.

이 코드는 주피터 노트북(.ipynb)을 생성하고, 위에서부터 아래로 총 5개의 셀에 차례대로 복사하여 실행할 수 있도록 분할되어 있습니다.

---

## [Cell 1] 라이브러리 및 학습 데이터 로드
데이터를 불러오고 기본적인 정보를 확인합니다. `DATA_PATH` 변수에는 교수님이 제공하는 학습 데이터 파일명을 입력합니다.

```python
# ==========================================
# [Step 1] 라이브러리 로드 및 데이터 로드
# ==========================================
import pandas as pd
import numpy as np
import itertools
import pickle
import time
import warnings
warnings.filterwarnings('ignore')  # 불필요한 경고 생략

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.base import clone

# === 데이터 파일 경로 (시험 현장에서 교수님이 준 CSV 파일명으로 변경) ===
DATA_PATH = 'creditcard.csv' 

print("데이터를 로드하고 있습니다...")
df = pd.read_csv(DATA_PATH)
print(f"✅ 데이터 로드 완료! 데이터 모양: {df.shape}")
print(f"✅ 클래스 분포:\n{df['Class'].value_counts()}")
print(f"✅ 사기 데이터 비율: {df['Class'].mean() * 100:.4f}%")
```

---

## [Cell 2] 평가 보조 함수 정의
교차 검증 시 임계값(Threshold)을 실시간으로 튜닝하기 위해, 빠른 연산용 Macro F1 함수와 임계값 탐색기를 선언합니다.

```python
# ==========================================
# [Step 2] 고속 채점 알고리즘 및 임계값 탐색기 정의
# ==========================================

# Numpy 기반 벡터화 연산으로 sklearn 대비 30배 빠른 Macro F1 계산기
def fast_f1_macro(y_true_bool, preds_bool):
    tp = np.sum(preds_bool & y_true_bool)
    fp = np.sum(preds_bool & ~y_true_bool)
    fn = np.sum(~preds_bool & y_true_bool)
    tn = np.sum(~preds_bool & ~y_true_bool)
    
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
    
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    
    return (f1_0 + f1_1) / 2.0

# 0.01부터 0.99까지 돌면서 최고의 Macro F1을 만드는 임계값(Threshold) 반환
def find_best_threshold(probs, y_true_bool, n_steps=199):
    best_score = -1.0
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, n_steps):
        preds_bool = (probs >= thresh)
        score = fast_f1_macro(y_true_bool, preds_bool)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score

print("✅ 평가 보조 함수 선언 완료!")
```

---

## [Cell 3] 피처 5개 조합 초고속 1차 스크리닝 (전수 탐색)
V1부터 V10 중 5개의 특성을 선택하는 총 252개 조합에 대해 전수 탐색을 실시합니다. 속도 단축을 위해 **정상 거래(Class 0) 데이터에서 10,000개만 무작위로 추출(Downsampling)**하여 3-Fold 교차 검증을 빠르게 실행합니다. (약 3~40초 소요)

```python
# ==========================================
# [Step 3] 252개 피처 조합 전수 고속 스크리닝 (3-Fold CV)
# ==========================================
features_pool = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']
target_col = 'Class'

X_all = df[features_pool].values
y_all = df[target_col].values

# 고속 스크리닝을 위한 데이터셋 샘플링 수행 (속도 극대화)
np.random.seed(42)
pos_mask = (y_all == 1)
neg_indices = np.where(~pos_mask)[0]
pos_indices = np.where(pos_mask)[0]

# 정상 데이터 10,000개만 랜덤 추출하여 사기 데이터와 결합
sample_neg_n = min(10000, len(neg_indices))
sampled_neg_idx = np.random.choice(neg_indices, size=sample_neg_n, replace=False)
screen_idx = np.concatenate([pos_indices, sampled_neg_idx])
np.random.shuffle(screen_idx)

X_screen = X_all[screen_idx]
y_screen = y_all[screen_idx]

combinations = list(itertools.combinations(range(10), 5))
print(f"총 {len(combinations)}개의 피처 조합을 전수 조사합니다... (다운샘플링 크기: {len(screen_idx)})")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
folds_screen = list(skf.split(X_screen, y_screen))

cv_results = []
t_start = time.time()

for idx, combo_idx in enumerate(combinations):
    combo_list = [features_pool[i] for i in combo_idx]
    X_combo = X_screen[:, combo_idx]
    
    fold_scores = []
    for train_idx, val_idx in folds_screen:
        X_tr, X_va = X_combo[train_idx], X_combo[val_idx]
        y_tr, y_va = y_screen[train_idx], y_screen[val_idx]
        y_va_bool = (y_va == 1)
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        
        # 50 trees를 가볍게 사용하는 ExtraTreesClassifier로 고속 선별
        et = ExtraTreesClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1)
        et.fit(X_tr_s, y_tr)
        probs = et.predict_proba(X_va_s)[:, 1]
        
        _, score = find_best_threshold(probs, y_va_bool, n_steps=29)
        fold_scores.append(score)
    
    mean_score = np.mean(fold_scores)
    cv_results.append({
        'features': combo_list,
        'combo_idx': combo_idx,
        'mean_f1': mean_score
    })
    
    if (idx + 1) % 50 == 0:
        elapsed = time.time() - t_start
        best_so_far = max(r['mean_f1'] for r in cv_results)
        print(f"  [{idx + 1}/252] 경과: {elapsed:.1f}초 | 현재 최고 스크리닝 F1: {best_so_far:.5f}")

# 정렬
cv_results.sort(key=lambda x: x['mean_f1'], reverse=True)

print(f"\n✅ 스크리닝 완료! (총 {time.time() - t_start:.1f}초 소요)")
print(">> 스크리닝 기준 상위 3개 조합:")
for i in range(3):
    r = cv_results[i]
    print(f"   {i+1}등: {r['features']} | F1: {r['mean_f1']:.5f}")
```

---

## [Cell 4] 상위 피처 조합 정밀 검증 및 최종 모델 학습/저장
걸러진 상위 3개 피처 조합에 대해 **전체 데이터(28만 행)** 및 **100 Estimators**를 적용하여 정밀하게 교차 검증을 돌리고, 최적 모델과 임계값을 도출한 후 `best_model.pkl`로 최종 저장합니다. (약 1분 소요)

```python
# ==========================================
# [Step 4] 정밀 교차 검증 및 최종 모델 학습/저장
# ==========================================
print(">> 상위 3개 조합에 대해 전체 데이터 정밀 평가를 시작합니다...")

skf_full = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
folds_full = list(skf_full.split(X_all, y_all))

top_combos = [(r['features'], r['combo_idx']) for r in cv_results[:3]]

best_overall_score = -1.0
best_overall_config = {}

for rank, (combo, combo_idx) in enumerate(top_combos):
    print(f"\n[{rank+1}등 피처 조합] {combo}")
    X_combo = X_all[:, combo_idx]
    
    model_defs = {
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'Ensemble(ET+RF)': VotingClassifier(
            estimators=[
                ('et', ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)),
                ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
            ],
            voting='soft', weights=[2, 1])
    }
    
    for model_name, model_template in model_defs.items():
        fold_scores = []
        fold_thresholds = []
        t0 = time.time()
        
        for fi, (train_idx, val_idx) in enumerate(folds_full):
            X_tr, X_va = X_combo[train_idx], X_combo[val_idx]
            y_tr, y_va = y_all[train_idx], y_all[val_idx]
            y_va_bool = (y_va == 1)
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            
            model = clone(model_template)
            model.fit(X_tr_s, y_tr)
            probs = model.predict_proba(X_va_s)[:, 1]
            
            thresh, score = find_best_threshold(probs, y_va_bool, n_steps=99)
            fold_scores.append(score)
            fold_thresholds.append(thresh)
        
        mean_f1 = np.mean(fold_scores)
        mean_thresh = np.mean(fold_thresholds)
        elapsed = time.time() - t0
        print(f"  - {model_name}: CV F1 = {mean_f1:.5f} (임계값 평균: {mean_thresh:.3f}) [{elapsed:.1f}초]")
        
        if mean_f1 > best_overall_score:
            best_overall_score = mean_f1
            best_overall_config = {
                'features': combo,
                'combo_idx': combo_idx,
                'model_name': model_name,
                'model_template': model_template,
                'mean_f1': mean_f1,
                'mean_threshold': mean_thresh
            }

# 최종 학습 정보 적용
best_features = best_overall_config['features']
best_template = best_overall_config['model_template']
best_threshold = best_overall_config['mean_threshold']

print(f"\n=======================================================")
print(f"🏆 최종 선정 피처 조합: {best_features}")
print(f"🏆 최종 선정 모델: {best_overall_config['model_name']}")
print(f"🏆 결정 임계값: {best_threshold:.3f} | CV F1: {best_overall_score:.5f}")
print(f"=======================================================")

# 전체 학습 데이터로 최종 학습
X_final = df[best_features]
y_final = df[target_col]

final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X_final)

final_model = clone(best_template)
print(f">> 전체 데이터({len(X_final)}건)로 최종 학습을 수행합니다...")
final_model.fit(X_final_scaled, y_final)

# 자체 평가
probs_all = final_model.predict_proba(X_final_scaled)[:, 1]
preds_all = (probs_all >= best_threshold).astype(int)
self_f1 = f1_score(y_final, preds_all, average='macro')
print(f"📌 전체 학습 데이터 자체 Macro F1: {self_f1:.4f}")
print(classification_report(y_final, preds_all))

# 모델 피클로 직렬화하여 디스크 저장
model_data = {
    'model': final_model,
    'features': best_features,
    'scaler': final_scaler,
    'threshold': best_threshold
}

with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print(f"\n✅ 최종 모델 'best_model.pkl' 저장 성공!")
```

---

## [Cell 5] 실시간 테스트 데이터 예측 및 채점 실행
생성된 `best_model.pkl` 파일을 로드하여 새로 제공받은 테스트 데이터를 채점 및 예측하고 결과를 `predictions.csv`로 저장하는 최종 예측 모듈입니다.

```python
# ==========================================
# [Step 5] 실시간 테스트 데이터 평가 및 채점 실행
# ==========================================
import os
import sys

model_path = 'best_model.pkl'
if not os.path.exists(model_path):
    print(f"❌ [오류] {model_path} 파일이 존재하지 않습니다. Step 4를 먼저 실행해 주세요.")
else:
    # 모델 로드
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data['model']
    selected_features = model_data['features']
    scaler = model_data['scaler']
    threshold = model_data.get('threshold', 0.5)
    
    print(f"✅ 모델 로드 성공! (알고리즘: {type(model).__name__})")
    print(f"📌 선정된 5개 피처: {selected_features}")
    print(f"📌 결정 임계값: {threshold:.3f}")
    
    # 채점용 테스트 CSV 경로 입력
    test_path = input("\n👉 채점용 테스트 CSV 파일 경로를 입력하세요: ").strip()
    test_path = test_path.strip("'\"")
    
    if not os.path.exists(test_path):
        print(f"❌ [오류] 파일 경로 '{test_path}'가 존재하지 않습니다.")
    else:
        try:
            try:
                test_df = pd.read_csv(test_path, encoding='utf-8')
            except UnicodeDecodeError:
                test_df = pd.read_csv(test_path, encoding='cp949')
                
            print(f"✅ 테스트 데이터 업로드 완료! shape: {test_df.shape}")
            
            # 피처 체크
            missing_features = [f for f in selected_features if f not in test_df.columns]
            if missing_features:
                print(f"❌ [오류] 테스트 데이터에 필수 피처 {missing_features}이 없습니다.")
            else:
                X_test = test_df[selected_features]
                
                # 스케일러 적용 및 예측
                X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test.values
                probs = model.predict_proba(X_test_scaled)[:, 1]
                predictions = (probs >= threshold).astype(int)
                
                # 예측값 저장
                output_df = pd.DataFrame({'prediction': predictions})
                output_df.to_csv('predictions.csv', index=False)
                print(f"✅ 예측 결과 파일 'predictions.csv' 생성 완료!")
                
                # 정답 라벨이 있으면 성능 지표 출력
                if 'Class' in test_df.columns:
                    y_test = test_df['Class']
                    macro_f1 = f1_score(y_test, predictions, average='macro')
                    
                    print("\n" + "="*50)
                    print("📊 기말고사 실시간 채점 결과 보고서 📊")
                    print("="*50)
                    print(classification_report(y_test, predictions))
                    print(f"⭐ 최종 Macro F1-Score: {macro_f1:.5f} ⭐")
                    print("="*50)
                else:
                    print("\n💡 [알림] 정답(Class) 컬럼이 없어 채점표는 생략하고 예측 결과 파일만 생성했습니다.")
        except Exception as e:
            print(f"❌ [오류] 실행 중 에러 발생: {e}")
```
