import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler

# ------------------------------
# 1) 設定路徑
# ------------------------------
DATA_DIR = "C:/Users/eric3/Downloads/40_初賽資料_V3 1/preliminary_data/"
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
RESULT_PATH = os.path.join(DATA_DIR, "result_CatBoost.csv")
PREDICT_ACCT_PATH = os.path.join(DATA_DIR, "acct_predict.csv")

# 讀取分類後交易檔案
alert_txn = pd.read_csv(os.path.join(OUTPUT_DIR, 'alert_transactions.csv'))
normal_txn = pd.read_csv(os.path.join(OUTPUT_DIR, 'normal_transactions.csv'))
predict_txn = pd.read_csv(os.path.join(OUTPUT_DIR, 'predict_transactions.csv'))

# 讀取官方預測帳戶清單（4780 個）
df_predict_accts = pd.read_csv(PREDICT_ACCT_PATH)
official_accts = set(df_predict_accts['acct'].astype(str))
print(f"官方待預測帳戶數: {len(official_accts)}")

# ------------------------------
# 2) 顯示三個檔案筆數與完整性
# ------------------------------
print(f"\n原始交易總筆數檢查:")
print(f"警示交易筆數: {len(alert_txn)}")
print(f"一般交易筆數: {len(normal_txn)}")
print(f"待預測交易筆數: {len(predict_txn)}")
total = len(alert_txn) + len(normal_txn) + len(predict_txn)
print(f"總計: {total}\n")

# ------------------------------
# 3) Transaction 特徵預處理
# ------------------------------
def preprocess_data(transaction_df):
    df = transaction_df.copy()
    # 時間欄位轉秒
    def time_to_seconds(time_str):
        if pd.isna(time_str) or time_str=='':
            return 0
        try:
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                if len(parts)==3:
                    h,m,s = parts
                    return int(h)*3600 + int(m)*60 + int(s)
                elif len(parts)==2:
                    h,m = parts
                    return int(h)*3600 + int(m)*60
            return float(time_str)
        except:
            return 0
    df['txn_time_seconds'] = df['txn_time'].apply(time_to_seconds)
    # 類別欄位 map
    df['from_acct_type'] = df['from_acct_type'].map({'01':1,'02':0}).fillna(0)
    df['to_acct_type'] = df['to_acct_type'].map({'01':1,'02':0}).fillna(0)
    df['is_self_txn'] = df['is_self_txn'].map({'Y':1,'N':0,'UNK':0}).fillna(0)
    channel_map = {'ATM':1,'臨櫃':2,'行動銀行(行銀)':3,'網路銀行(網銀)':4,'語音':5,
                   'eATM':6,'電子支付':7,'系統排程交易':99,'UNK':0}
    df['channel_type'] = df['channel_type'].map(channel_map).fillna(0)
    currency_map = {'TWD':1,'USD':2,'JPY':3,'CNY':4,'HKD':5,'EUR':6,'GBP':7,'AUD':8,'CAD':9,'SGD':10}
    df['currency_type'] = df['currency_type'].map(currency_map).fillna(0)
    return df

def create_transaction_features(transaction_df):
    df = preprocess_data(transaction_df)
    # 匯款方特徵
    from_feat_list = []
    for acct, group in tqdm(df.groupby('from_acct'), desc="Processing from_acct"):
        d = {
            'acct': acct,
            'from_txn_count': len(group),
            'from_txn_amt_sum': group['txn_amt'].sum(),
            'from_txn_amt_mean': group['txn_amt'].mean(),
            'from_txn_amt_std': group['txn_amt'].std(),
            'from_txn_amt_min': group['txn_amt'].min(),
            'from_txn_amt_max': group['txn_amt'].max(),
            'from_txn_time_mean': group['txn_time_seconds'].mean(),
            'from_txn_time_std': group['txn_time_seconds'].std(),
            'from_self_txn_count': group['is_self_txn'].sum(),
            'from_unique_channels': group['channel_type'].nunique(),
            'from_channel_mean': group['channel_type'].mean(),
            'from_unique_currencies': group['currency_type'].nunique(),
            'from_esun_ratio': group['from_acct_type'].mean(),
            'to_esun_ratio_from': group['to_acct_type'].mean()
        }
        from_feat_list.append(d)
    from_feat = pd.DataFrame(from_feat_list)
    # 收款方特徵
    to_feat_list = []
    for acct, group in tqdm(df.groupby('to_acct'), desc="Processing to_acct"):
        d = {
            'acct': acct,
            'to_txn_count': len(group),
            'to_txn_amt_sum': group['txn_amt'].sum(),
            'to_txn_amt_mean': group['txn_amt'].mean(),
            'to_txn_amt_std': group['txn_amt'].std(),
            'to_txn_amt_min': group['txn_amt'].min(),
            'to_txn_amt_max': group['txn_amt'].max(),
            'to_txn_time_mean': group['txn_time_seconds'].mean(),
            'to_txn_time_std': group['txn_time_seconds'].std(),
            'to_self_txn_count': group['is_self_txn'].sum(),
            'to_unique_channels': group['channel_type'].nunique(),
            'to_channel_mean': group['channel_type'].mean(),
            'to_unique_currencies': group['currency_type'].nunique(),
            'to_esun_ratio': group['to_acct_type'].mean(),
            'from_esun_ratio_to': group['from_acct_type'].mean()
        }
        to_feat_list.append(d)
    to_feat = pd.DataFrame(to_feat_list)
    feat = pd.merge(from_feat, to_feat, on='acct', how='outer').fillna(0)
    # 衍生特徵
    feat['total_txn_count'] = feat['from_txn_count'] + feat['to_txn_count']
    feat['total_txn_amt'] = feat['from_txn_amt_sum'] + feat['to_txn_amt_sum']
    feat['net_flow'] = feat['from_txn_amt_sum'] - feat['to_txn_amt_sum']
    feat['self_txn_ratio'] = (feat['from_self_txn_count'] + feat['to_self_txn_count']) / feat['total_txn_count'].replace(0,1)
    feat['avg_txn_amt'] = feat['total_txn_amt'] / feat['total_txn_count'].replace(0,1)
    feat['from_to_ratio'] = feat['from_txn_count'] / feat['to_txn_count'].replace(0,1)
    feat['amt_std_mean_ratio'] = (feat['from_txn_amt_std'] + feat['to_txn_amt_std']) / (feat['from_txn_amt_mean'] + feat['to_txn_amt_mean']).replace(0,1)
    return feat

# ------------------------------
# 4) 生成交易特徵
# ------------------------------
print("生成特徵中...")
alert_features = create_transaction_features(alert_txn)
normal_features = create_transaction_features(normal_txn)
predict_features = create_transaction_features(predict_txn)

print(f"alert_features 帳戶數: {len(alert_features)}")
print(f"normal_features 帳戶數: {len(normal_features)}")
print(f"predict_features 帳戶數: {len(predict_features)}")

# ------------------------------
# 5) 蒙地卡羅式迭代訓練與預測
# ------------------------------
n_iterations = 20
predict_probas = []

for i in range(n_iterations):
    print(f"\nMonte Carlo Iteration {i+1}/{n_iterations}")
    # 欠抽樣
    train_features = pd.concat([alert_features, normal_features], axis=0).reset_index(drop=True)
    train_features['label'] = [1]*len(alert_features) + [0]*len(normal_features)
    X = train_features.drop(columns=['acct','label'])
    y = train_features['label']
    
    rus = RandomUnderSampler(random_state=i, sampling_strategy=0.3)
    X_res, y_res = rus.fit_resample(X, y)
    
    # Low variance
    selector = VarianceThreshold(threshold=0.01)
    X_res = pd.DataFrame(selector.fit_transform(X_res), columns=X.columns[selector.get_support()])
    
    # 分割訓練/驗證
    X_tr, X_val, y_tr, y_val = train_test_split(X_res, y_res, test_size=0.3, random_state=i, stratify=y_res)
    
    # CatBoost 訓練
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.01,
        depth=12,
        l2_leaf_reg=6,
        random_seed=i,
        verbose=300,
        scale_pos_weight=10,
        eval_metric='AUC',
        early_stopping_rounds=None,
        thread_count=-1,
        bootstrap_type='Bayesian',
        loss_function='Logloss'
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    
    # 預測 predict
    X_pred = predict_features.drop(columns=['acct'])
    X_pred = pd.DataFrame(selector.transform(X_pred), columns=X_pred.columns[selector.get_support()])
    proba = model.predict_proba(X_pred)[:,1]
    predict_probas.append(proba)

# ------------------------------
# 6) 指數放大 + 正規化 + 幾何平均收斂
# ------------------------------
predict_probas_exp = []
for p in predict_probas:
    safe_p = np.clip(p, 0, 1)
    exp_p = np.exp(safe_p)
    predict_probas_exp.append(exp_p)
predict_probas_exp = np.array(predict_probas_exp)

exp_probas_mean = np.mean(predict_probas_exp, axis=0)
exp_probas_norm = (exp_probas_mean - exp_probas_mean.min()) / (exp_probas_mean.max() - exp_probas_mean.min() + 1e-8)

# 加入 predict_features
predict_features['prob_exp'] = exp_probas_norm

# 幾何平均收斂（你原本的邏輯）
acct_scores = predict_features.groupby('acct')['prob_exp'].agg(lambda x: x.prod()**(1/len(x))).reset_index()
acct_scores = acct_scores.rename(columns={'prob_exp': 'geom_score'})

# ------------------------------
# 7) 對齊官方 4780 個帳戶（保留原始順序！）
# ------------------------------
# 關鍵修改：直接複製 acct_predict.csv 的順序
result_df = df_predict_accts.copy()  # 保留原始 acct 順序
result_df['acct'] = result_df['acct'].astype(str)
result_df['label'] = 0  # 預設全為 0

# 合併你的預測分數
result_df = result_df.merge(acct_scores, on='acct', how='left')
result_df['geom_score'] = result_df['geom_score'].fillna(0)  # 無交易帳戶 → 0

# ------------------------------
# 8) 標記 Top 250 警示帳號
# ------------------------------
top_n = 250
top_idx = result_df['geom_score'].nlargest(top_n).index
result_df['label'] = 0
result_df.loc[top_idx, 'label'] = 1

# ------------------------------
# 9) 輸出 result.csv 並驗證
# ------------------------------
result_df[['acct', 'label']].to_csv(RESULT_PATH, index=False)

print(f"\n完成，已生成 result.csv")
print(f"預測總帳號數: {len(result_df)}")
print(f"標記為警示帳號數: {result_df['label'].sum()}")
print(f"標記為正常帳號數: {len(result_df) - result_df['label'].sum()}")

# ------------------------------
# 10) 最終驗證：與 acct_predict.csv 完全一致
# ------------------------------
df_check = pd.read_csv(RESULT_PATH)
official_check = pd.read_csv(PREDICT_ACCT_PATH)

missing = set(official_check['acct'].astype(str)) - set(df_check['acct'].astype(str))
extra = set(df_check['acct'].astype(str)) - set(official_check['acct'].astype(str))

print(f"\n驗證結果：")
print(f"官方帳戶數: {len(official_check)}")
print(f"結果帳戶數: {len(df_check)}")
print(f"遺漏帳戶: {len(missing)} 個" + (f" → {list(missing)[:5]}..." if missing else ""))
print(f"多餘帳戶: {len(extra)} 個" + (f" → {list(extra)[:5]}..." if extra else ""))
print(f"是否完全一致: {'Yes' if len(missing)==0 and len(extra)==0 else 'No'}")

if len(missing) == 0 and len(extra) == 0:
    print("結果與官方帳戶完全對齊！可提交！")
else:
    print("請檢查資料完整性！")