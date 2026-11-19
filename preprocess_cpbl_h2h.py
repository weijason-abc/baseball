# -*- coding: utf-8 -*-
"""CPBL 2026 逐打席四分類資料前處理（含打者、投手、H2H 歷史特徵）。

可單獨執行本檔案檢查資料；訓練時由 train_cpbl_mlp_h2h.py 匯入 prepare_data。
前處理只使用每場比賽日期「之前」的打席統計，同一天的打席不納入歷史特徵。
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
PA_FILE = ROOT / "rebas_CPBL_2026_六隊逐打席_修正版.xlsx"

# 類別編號必須與訓練程式的 Dense(4, softmax) 一致。
LABELS = ["出局", "一安", "長打", "保送"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

# 對尚無歷史對戰紀錄者，使用固定的平滑先驗（不會從未來資料估算）。
PRIOR_STRENGTH = 20.0
PRIOR_RATES = {"out": 0.66, "single": 0.19, "xbh": 0.05, "walk": 0.10}

CATEGORICAL_COLUMNS = ["打者", "投手", "上下半局", "進攻球隊", "防守球隊"]
NUMERIC_COLUMNS = [
    "局數", "打席前出局數", "打席前一壘有人", "打席前二壘有人", "打席前三壘有人",
    "打席前客隊分", "打席前主隊分",
    "b_prior_pa_count", "b_out_rate", "b_single_rate", "b_xbh_rate", "b_walk_rate",
    "p_prior_pa_count", "p_out_rate", "p_single_rate", "p_xbh_rate", "p_walk_rate",
    "h2h_prior_pa_count", "h2h_out_rate", "h2h_single_rate", "h2h_xbh_rate", "h2h_walk_rate",
]


def classify_result(value):
    """回傳 (標籤, 排除原因)；保送包含四壞、故意四壞與觸身球。"""
    if pd.isna(value) or not str(value).strip():
        return None, "other"

    text = str(value).strip().split(",", 1)[0].strip()

    # 依專題定義排除失誤、犧牲飛球、短打、野選及其他非四分類事件。
    if "犧牲飛球" in text or "短打" in text:
        return None, "sacrifices_bunts"
    if "失誤" in text or "(E)" in text or "（E）" in text:
        return None, "errors"
    if "野選" in text or "妨礙" in text:
        return None, "other"

    if text.startswith("一安"):
        return "一安", None
    if text.startswith(("二安", "三安", "全壘打")):
        return "長打", None
    if "四壞" in text or "觸身球" in text:
        return "保送", None
    if "出局" in text or "三振" in text or "雙殺" in text:
        return "出局", None

    return None, "other"


def add_prior_stats(df, keys, prefix):
    """產生只包含之前日期的打席數與四分類機率，避免當日及未來資料洩漏。"""
    source_columns = ["is_out", "is_single", "is_xbh", "is_walk", "pa_count"]
    daily = (
        df.groupby(keys + ["日期"], dropna=False)[source_columns]
        .sum()
        .reset_index()
        .sort_values(keys + ["日期"])
    )

    grouped = daily.groupby(keys, dropna=False)
    for col in source_columns:
        daily[prefix + "prior_" + col] = (
            grouped[col].cumsum() - daily[col]
        ).astype("float32")

    n = daily[prefix + "prior_pa_count"]
    features = [prefix + "prior_pa_count"]
    for name, prior_rate in PRIOR_RATES.items():
        col = prefix + name + "_rate"
        daily[col] = (
            daily[prefix + "prior_is_" + name] + PRIOR_STRENGTH * prior_rate
        ) / (n + PRIOR_STRENGTH)
        features.append(col)

    return df.merge(
        daily[keys + ["日期"] + features],
        on=keys + ["日期"],
        how="left",
        validate="many_to_one",
    )


def prepare_data(pa_file=PA_FILE):
    """讀取 Excel、建立四分類與歷史特徵，依日期分為訓練／驗證／測試。"""
    pa_file = Path(pa_file)
    if not pa_file.is_file():
        raise FileNotFoundError(
            f"找不到逐打席 Excel：{pa_file}\n"
            "請將 rebas_CPBL_2026_六隊逐打席_修正版.xlsx 放在兩個 .py 的同一資料夾。"
        )

    df = pd.read_excel(pa_file, sheet_name="逐打席", dtype={"比賽識別碼": "string"})
    required = [
        "比賽日期", "比賽識別碼", "全場第幾打席", "打者", "投手", "打擊結果",
        *CATEGORICAL_COLUMNS[2:],
        "局數", "打席前出局數", "打席前一壘有人", "打席前二壘有人",
        "打席前三壘有人", "打席前客隊分", "打席前主隊分",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Excel 的『逐打席』工作表缺少欄位：{missing}")

    raw_rows = len(df)
    df["日期"] = pd.to_datetime(df["比賽日期"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["日期", "比賽識別碼", "全場第幾打席"]).copy()
    df = df.drop_duplicates(subset=["比賽識別碼", "全場第幾打席"]).copy()

    # 分類前先算排除筆數，避免將未納入分類的打席混進歷史比率。
    classified = df["打擊結果"].map(classify_result)
    df["標籤"] = classified.map(lambda x: x[0])
    reasons = classified.map(lambda x: x[1])
    ignored_errors = int((reasons == "errors").sum())
    ignored_sacrifices_bunts = int((reasons == "sacrifices_bunts").sum())
    ignored_other = int((reasons == "other").sum())
    df = df[df["標籤"].notna()].copy()

    if df.empty:
        raise ValueError("四分類後沒有可用的打席資料，請檢查『打擊結果』欄位。")

    df["label_id"] = df["標籤"].map(LABEL_TO_ID).astype("int32")
    for name, label in [("out", "出局"), ("single", "一安"),
                        ("xbh", "長打"), ("walk", "保送")]:
        df["is_" + name] = (df["標籤"] == label).astype("int32")
    df["pa_count"] = 1

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string").fillna("未知").str.strip()
        df[col] = df[col].replace("", "未知").astype(str)

    # 打者、投手、對戰組合，皆只取之前日期之歷史結果。
    df = add_prior_stats(df, ["打者"], "b_")
    df = add_prior_stats(df, ["投手"], "p_")
    df = add_prior_stats(df, ["打者", "投手"], "h2h_")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dates = sorted(df["日期"].unique())
    if len(dates) < 10:
        raise ValueError("可用比賽日期少於 10 天，無法依時間建立訓練／驗證／測試資料集。")

    cut_a = int(len(dates) * 0.70)
    cut_b = int(len(dates) * 0.85)
    train = df[df["日期"].isin(dates[:cut_a])].copy()
    valid = df[df["日期"].isin(dates[cut_a:cut_b])].copy()
    test = df[df["日期"].isin(dates[cut_b:])].copy()

    # 使用訓練集 fit；驗證／測試集只 transform，避免前處理資訊洩漏。
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLUMNS),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_COLUMNS),
        ]
    )
    selected = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    X_train = preprocessor.fit_transform(train[selected]).astype("float32")
    X_valid = preprocessor.transform(valid[selected]).astype("float32")
    X_test = preprocessor.transform(test[selected]).astype("float32")

    y_train = train["label_id"].to_numpy(dtype="int32")
    y_valid = valid["label_id"].to_numpy(dtype="int32")
    y_test = test["label_id"].to_numpy(dtype="int32")

    print(f"原始資料：{raw_rows:,} 筆；四分類有效資料：{len(df):,} 筆")
    print(f"排除：失誤 {ignored_errors:,}、犧牲飛球／短打 {ignored_sacrifices_bunts:,}、其他 {ignored_other:,}")
    print(f"資料集：訓練 {len(train):,}／驗證 {len(valid):,}／測試 {len(test):,}")
    print(f"輸入特徵：{X_train.shape[1]} 維（類別經 One-Hot、數值經標準化）")
    print("有效打席類別：", {name: int((df["標籤"] == name).sum()) for name in LABELS})

    return {
        "train": train, "valid": valid, "test": test,
        "X_train": X_train, "X_valid": X_valid, "X_test": X_test,
        "y_train": y_train, "y_valid": y_valid, "y_test": y_test,
        "prep": preprocessor,
        "cats": CATEGORICAL_COLUMNS,
        "nums": NUMERIC_COLUMNS,
        "labels": LABELS,
        "original_rows": raw_rows,
        "valid_rows": len(df),
        "ignored_errors": ignored_errors,
        "ignored_sacrifices_bunts": ignored_sacrifices_bunts,
        "ignored_other": ignored_other,
    }


if __name__ == "__main__":
    prepare_data()
