import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# 讀取資料
dir_path = "../src/"
transaction_df = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
alert_df = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
predict_accounts_df = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))

# 顯示資料基本資訊
print("資料讀取完成:")
print(f"交易紀錄筆數: {len(transaction_df)}")
print(f"警示帳戶數: {len(alert_df)}")
print(f"待預測帳戶數: {len(predict_accounts_df)}")

# 資料預處理函數
def preprocess_data(transaction_df, alert_df):
    # 複製資料避免修改原始資料
    df = transaction_df.copy()
    
    # 處理時間欄位 - 將時間字串轉換為數值
    def time_to_seconds(time_str):
        if pd.isna(time_str) or time_str == '':
            return 0
        try:
            # 處理時間格式 (HH:MM:SS)
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                if len(parts) == 3:
                    hours, minutes, seconds = parts
                    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                elif len(parts) == 2:
                    hours, minutes = parts
                    return int(hours) * 3600 + int(minutes) * 60
            # 如果已經是數值，直接返回
            return float(time_str)
        except:
            return 0
    
    # 轉換交易時間
    df['txn_time_seconds'] = df['txn_time'].apply(time_to_seconds)
    
    # 處理類別型變數
    df['from_acct_type'] = df['from_acct_type'].map({'01': 1, '02': 0}).fillna(0)
    df['to_acct_type'] = df['to_acct_type'].map({'01': 1, '02': 0}).fillna(0)
    df['is_self_txn'] = df['is_self_txn'].map({'Y': 1, 'N': 0, 'UNK': 0}).fillna(0)
    
    # 處理通路類型 - 使用提供的代碼表
    channel_mapping = {
        'ATM': 1, '臨櫃': 2, '行動銀行(行銀)': 3, '網路銀行(網銀)': 4,
        '語音': 5, 'eATM': 6, '電子支付': 7, '系統排程交易': 99, 'UNK': 0
    }
    df['channel_type'] = df['channel_type'].map(channel_mapping).fillna(0)
    
    # 處理幣別 - 主要幣別轉換
    currency_mapping = {
        'TWD': 1, 'USD': 2, 'JPY': 3, 'CNY': 4, 'HKD': 5,
        'EUR': 6, 'GBP': 7, 'AUD': 8, 'CAD': 9, 'SGD': 10
    }
    df['currency_type'] = df['currency_type'].map(currency_mapping).fillna(0)
    
    return df

# 特徵工程函數
def create_features(transaction_df):
    df = preprocess_data(transaction_df, None)
    
    # 建立特徵 - 從匯款方角度
    from_features = df.groupby('from_acct').agg({
        'txn_amt': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'txn_time_seconds': ['mean', 'std'],
        'is_self_txn': 'sum',
        'channel_type': ['nunique', 'mean'],
        'currency_type': 'nunique',
        'from_acct_type': 'mean',
        'to_acct_type': 'mean'
    }).reset_index()
    
    # 扁平化欄位名稱
    from_features.columns = [
        'acct', 'from_txn_count', 'from_txn_amt_sum', 'from_txn_amt_mean',
        'from_txn_amt_std', 'from_txn_amt_min', 'from_txn_amt_max',
        'from_txn_time_mean', 'from_txn_time_std', 'from_self_txn_count',
        'from_unique_channels', 'from_channel_mean', 'from_unique_currencies',
        'from_esun_ratio', 'to_esun_ratio_from'
    ]
    
    # 建立特徵 - 從收款方角度
    to_features = df.groupby('to_acct').agg({
        'txn_amt': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'txn_time_seconds': ['mean', 'std'],
        'is_self_txn': 'sum',
        'channel_type': ['nunique', 'mean'],
        'currency_type': 'nunique',
        'to_acct_type': 'mean',
        'from_acct_type': 'mean'
    }).reset_index()
    
    to_features.columns = [
        'acct', 'to_txn_count', 'to_txn_amt_sum', 'to_txn_amt_mean',
        'to_txn_amt_std', 'to_txn_amt_min', 'to_txn_amt_max',
        'to_txn_time_mean', 'to_txn_time_std', 'to_self_txn_count',
        'to_unique_channels', 'to_channel_mean', 'to_unique_currencies',
        'to_esun_ratio', 'from_esun_ratio_to'
    ]
    
    # 合併特徵
    features = pd.merge(from_features, to_features, on='acct', how='outer')
    
    # 處理缺失值
    features = features.fillna(0)
    
    # 計算衍生特徵
    features['total_txn_count'] = features['from_txn_count'] + features['to_txn_count']
    features['total_txn_amt'] = features['from_txn_amt_sum'] + features['to_txn_amt_sum']
    features['net_flow'] = features['from_txn_amt_sum'] - features['to_txn_amt_sum']
    features['self_txn_ratio'] = (features['from_self_txn_count'] + features['to_self_txn_count']) / features['total_txn_count'].replace(0, 1)
    
    # 添加更多衍生特徵
    features['avg_txn_amt'] = features['total_txn_amt'] / features['total_txn_count'].replace(0, 1)
    features['from_to_ratio'] = features['from_txn_count'] / features['to_txn_count'].replace(0, 1)
    features['amt_std_mean_ratio'] = (features['from_txn_amt_std'] + features['to_txn_amt_std']) / (features['from_txn_amt_mean'] + features['to_txn_amt_mean']).replace(0, 1)
    
    return features

# 建立特徵資料集
print("\n正在建立特徵...")
features_df = create_features(transaction_df)

# 合併特徵與標籤資料
print("正在合併特徵與標籤...")
alert_df['label'] = 1  # 所有警示帳戶標記為1

# 取得所有出現過的帳戶
all_from_accounts = transaction_df['from_acct'].unique()
all_to_accounts = transaction_df['to_acct'].unique()
all_accounts = np.unique(np.concatenate([all_from_accounts, all_to_accounts]))

# 建立所有帳戶的基礎資料集
all_accounts_df = pd.DataFrame({'acct': all_accounts})

# 合併特徵和標籤
model_df = all_accounts_df.merge(features_df, on='acct', how='left')
model_df = model_df.merge(alert_df[['acct', 'label']], on='acct', how='left')

# 將非警示帳戶標記為0
model_df['label'] = model_df['label'].fillna(0)

# 處理缺失值 - 用0填充數值特徵
numeric_columns = model_df.select_dtypes(include=[np.number]).columns
model_df[numeric_columns] = model_df[numeric_columns].fillna(0)

print(f"特徵資料集形狀: {model_df.shape}")
print(f"警示帳戶數量: {model_df['label'].sum()}")
print(f"非警示帳戶數量: {len(model_df) - model_df['label'].sum()}")

# 準備特徵和標籤
feature_columns = [col for col in model_df.columns if col not in ['acct', 'label']]
X = model_df[feature_columns]
y = model_df['label']

print(f"特徵數量: {len(feature_columns)}")

# 檢查特徵是否有無限值或NaN
if X.isnull().any().any():
    print("警告: 特徵中包含缺失值")
    X = X.fillna(0)

if np.isinf(X.values).any():
    print("警告: 特徵中包含無限值")
    X = X.replace([np.inf, -np.inf], 0)

# 處理極端不平衡資料 - 使用 undersampling
print("\n處理資料不平衡...")
rus = RandomUnderSampler(random_state=42, sampling_strategy=0.002)  # 負樣本:正樣本 = 100:1
X_resampled, y_resampled = rus.fit_resample(X, y)

print(f"重採樣後訓練集大小: {X_resampled.shape}")
print(f"重採樣後正樣本數量: {y_resampled.sum()}")
print(f"重採樣後負樣本數量: {len(y_resampled) - y_resampled.sum()}")

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

print(f"訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# 建立改進的 CatBoost 模型
catboost_model = CatBoostClassifier(
    iterations=500,            # 減少迭代次數
    learning_rate=0.05,        # 降低學習率
    depth=8,                   # 增加樹的深度
    l2_leaf_reg=5,             # 增加 L2 正則化
    random_seed=42,
    verbose=50,                # 每50次迭代顯示進度
    scale_pos_weight=5,        # 手動調整正樣本權重
    eval_metric='F1',          # 使用 F1 分數作為評估指標
    early_stopping_rounds=30,  # 早停法
    thread_count=-1,           # 使用所有可用的 CPU 核心
    bootstrap_type='Bayesian', # 使用貝葉斯 bootstrap
    loss_function='Logloss'    # 使用對數損失
)

# 訓練模型
print("正在訓練 CatBoost 模型...")
catboost_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    plot=False,
    use_best_model=True
)

# 預測
y_pred = catboost_model.predict(X_test)
y_pred_proba = catboost_model.predict_proba(X_test)[:, 1]

# 計算 F1 分數和其他指標
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = catboost_model.score(X_test, y_test)
auc_score = roc_auc_score(y_test, y_pred_proba)

# 評估模型
print("\n" + "="*60)
print("改進的 CatBoost 模型評估")
print("="*60)
print(f"F1 Score: {f1:.4f}")
print(f"Precision (精確率): {precision:.4f}")
print(f"Recall (召回率): {recall:.4f}")
print(f"Accuracy (準確率): {accuracy:.4f}")
print(f"AUC Score: {auc_score:.4f}")

print("\n混淆矩陣:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}")

print("\n詳細分類報告:")
print(classification_report(y_test, y_pred, digits=4))

# 顯示特徵重要性
feature_importance = catboost_model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n特徵重要性 (前20名):")
print("="*50)
print(importance_df.head(20))

# 在完整測試集上評估（使用原始不平衡資料）
print("\n" + "="*60)
print("在完整測試集上評估模型")
print("="*60)

# 使用原始測試集（保持原始分佈）
_, X_test_full, _, y_test_full = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

y_pred_full = catboost_model.predict(X_test_full)
y_pred_proba_full = catboost_model.predict_proba(X_test_full)[:, 1]

f1_full = f1_score(y_test_full, y_pred_full)
precision_full = precision_score(y_test_full, y_pred_full)
recall_full = recall_score(y_test_full, y_pred_full)

print(f"完整測試集 F1 Score: {f1_full:.4f}")
print(f"完整測試集 Precision: {precision_full:.4f}")
print(f"完整測試集 Recall: {recall_full:.4f}")

# 對指定帳戶進行預測
print("\n" + "="*60)
print("對指定帳戶進行預測")
print("="*60)

# 取得待預測帳戶列表
target_accounts = predict_accounts_df['acct'].tolist()

# 從特徵資料集中篩選出待預測帳戶的特徵
target_features = model_df[model_df['acct'].isin(target_accounts)]

if len(target_features) == 0:
    print("警告: 在交易紀錄中找不到待預測帳戶的資料")
    final_predictions = pd.DataFrame({'acct': target_accounts, 'label': 0})
else:
    # 準備預測特徵
    X_target = target_features[feature_columns]
    
    # 進行預測
    target_predictions = catboost_model.predict(X_target)
    target_probabilities = catboost_model.predict_proba(X_target)[:, 1]
    
    # 使用較高的閾值來提高精確率
    threshold = 0.7  # 提高預測閾值
    target_predictions_adj = (target_probabilities >= threshold).astype(int)
    
    # 建立預測結果
    prediction_results = target_features[['acct']].copy()
    prediction_results['label'] = target_predictions_adj
    prediction_results['probability'] = target_probabilities
    
    # 確保所有目標帳戶都有預測結果
    all_predictions = pd.DataFrame({'acct': target_accounts})
    final_predictions = all_predictions.merge(prediction_results, on='acct', how='left')
    
    # 對於在交易紀錄中沒出現的帳戶，設為非警示帳戶
    final_predictions['label'] = final_predictions['label'].fillna(0)
    final_predictions['probability'] = final_predictions['probability'].fillna(0)

# 顯示預測結果
print("\n預測結果:")
print("="*40)

# 統計結果
print(f"\n預測統計:")
print(f"總預測帳戶數: {len(final_predictions)}")
print(f"預測為警示帳戶: {final_predictions['label'].sum()}")
print(f"預測為非警示帳戶: {len(final_predictions) - final_predictions['label'].sum()}")

# # 顯示高風險帳戶（概率>0.5）
# high_risk_accounts = final_predictions[final_predictions['probability'] > 0.5]
# if len(high_risk_accounts) > 0:
#     print(f"\n高風險帳戶 (概率 > 0.5):")
#     for _, row in high_risk_accounts.iterrows():
#         print(f"  帳戶 {row['acct']}: 風險概率 {row['probability']:.4f}")

# 儲存預測結果
output_file = "account_predictions_catboost_improved.csv"
final_predictions[['acct', 'label']].to_csv(output_file, index=False)
print(f"\n預測結果已儲存至: {output_file}")

# 模型效能總結
print("\n" + "="*60)
print("改進的 CatBoost 模型效能總結")
print("="*60)
print(f"主要評估指標: F1 Score = {f1:.4f}")
print(f"精確率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"完整測試集 F1 Score: {f1_full:.4f}")

if f1_full > 0.5:
    print("✓ 模型表現良好")
elif f1_full > 0.3:
    print("○ 模型表現一般")
else:
    print("△ 模型需要進一步改進")

print(f"\nCatBoost 模型參數:")
print(f"最佳迭代次數: {catboost_model.get_best_iteration()}")
print(f"使用的特徵數量: {len(feature_columns)}")

print("\n任務完成！")