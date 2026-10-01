import pandas as pd
import os

# === 檔案路徑設定 ===
base_path = r"C:\Users\eric3\Downloads\40_初賽資料_V3 1\preliminary_data"
txn_file = os.path.join(base_path, "acct_transaction.csv")
alert_file = os.path.join(base_path, "acct_alert.csv")
predict_file = os.path.join(base_path, "acct_predict.csv")

# === 讀取資料 ===
txn_df = pd.read_csv(txn_file)
alert_df = pd.read_csv(alert_file)
predict_df = pd.read_csv(predict_file)

# === 確認欄位名稱 ===
alert_col = alert_df.columns[0]
predict_col = predict_df.columns[0]
print(f"📘 警示帳戶欄位名稱為: {alert_col}")
print(f"📘 預測帳戶欄位名稱為: {predict_col}")

# === 建立帳號集合 ===
alert_accts = set(alert_df[alert_col].astype(str))
predict_accts = set(predict_df[predict_col].astype(str))

# === 判斷交易類別 ===
txn_df["from_acct"] = txn_df["from_acct"].astype(str)
txn_df["to_acct"] = txn_df["to_acct"].astype(str)

txn_df["is_alert_txn"] = txn_df["from_acct"].isin(alert_accts) | txn_df["to_acct"].isin(alert_accts)
txn_df["is_predict_txn"] = txn_df["from_acct"].isin(predict_accts) | txn_df["to_acct"].isin(predict_accts)

# === 分類資料 ===
alert_txn_df = txn_df[txn_df["is_alert_txn"]].copy()
predict_txn_df = txn_df[~txn_df["is_alert_txn"] & txn_df["is_predict_txn"]].copy()
normal_txn_df = txn_df[~txn_df["is_alert_txn"] & ~txn_df["is_predict_txn"]].copy()

# === 匯出資料 ===
out_dir = os.path.join(base_path, "output")
os.makedirs(out_dir, exist_ok=True)

alert_txn_df.to_csv(os.path.join(out_dir, "alert_transactions.csv"), index=False)
predict_txn_df.to_csv(os.path.join(out_dir, "predict_transactions.csv"), index=False)
normal_txn_df.to_csv(os.path.join(out_dir, "normal_transactions.csv"), index=False)

# === 檢查筆數與比例 ===
total_original = len(txn_df)
total_split = len(alert_txn_df) + len(predict_txn_df) + len(normal_txn_df)

print("\n📊 === 數據完整性檢查 ===")
print(f"原始交易總筆數: {total_original}")
print(f"三份檔案加總筆數: {total_split}")
print(f"警示交易筆數: {len(alert_txn_df):,} ({len(alert_txn_df)/total_original*100:.2f}%)")
print(f"預測交易筆數: {len(predict_txn_df):,} ({len(predict_txn_df)/total_original*100:.2f}%)")
print(f"一般交易筆數: {len(normal_txn_df):,} ({len(normal_txn_df)/total_original*100:.2f}%)")

if total_original == total_split:
    print("✅ 筆數完全一致，資料切分正確！")
else:
    diff = total_split - total_original
    print(f"⚠️ 注意：筆數不符，差異 {diff:+,} 筆。請檢查是否有重複或遺漏。")

print("\n🎯 三份資料輸出與驗證完成！")
