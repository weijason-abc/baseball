from pathlib import Path
import re
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

LABELS = ['出局', '一安', '長打', '保送']
STATS = ['is_out', 'is_single', 'is_xbh', 'is_walk', 'pa_count']
PRIOR_RATES = [('out', 0.68), ('single', 0.16), ('xbh', 0.08), ('walk', 0.08)]


def excluded_reason(value):
    text = str(value).strip()
    if re.search(r'失誤|失策', text):
        return 'error'
    if re.search(r'犧牲|犧飛|犧打|短打|觸擊', text):
        return 'sacrifice_or_bunt'
    return None


def outcome(value):
    text = str(value).strip()
    if not text or excluded_reason(text):
        return None
    result = re.split(r'[,，]', text, maxsplit=1)[0].strip()
    if re.match(r'^(一安|一壘安打)', result):
        return '一安'
    if re.match(r'^(二安|三安|全壘打|二壘安打|三壘安打|全壘安打|場內全壘打)', result):
        return '長打'
    if any(word in result for word in ('四壞', '保送', '敬遠')):
        return '保送'
    if any(word in result for word in ('出局', '三振', '雙殺', '三殺', '接殺')):
        return '出局'
    return None


def prior_stats(df, person_col, prefix):
    counts = (
        df.groupby([person_col, '日期'], dropna=False)[STATS]
        .sum()
        .reset_index()
        .sort_values([person_col, '日期'])
    )
    grouped = counts.groupby(person_col, dropna=False)
    for name in STATS:
        counts[prefix + 'prior_' + name] = (
            grouped[name].cumsum() - counts[name]
        ).astype('float32')
    total = counts[prefix + 'prior_pa_count']
    for kind, base in PRIOR_RATES:
        counts[prefix + kind + '_rate'] = (
            counts[prefix + 'prior_is_' + kind] + 20 * base
        ) / (total + 20)
    keep = [person_col, '日期', prefix + 'prior_pa_count'] + [
        prefix + kind + '_rate' for kind, _ in PRIOR_RATES
    ]
    return df.merge(
        counts[keep], on=[person_col, '日期'], how='left', validate='many_to_one'
    )


def matchup_prior_stats(df):
    keys = ['打者', '投手']
    daily = (
        df.groupby(keys + ['日期'], dropna=False)[STATS]
        .sum()
        .reset_index()
        .sort_values(keys + ['日期'])
    )
    grouped = daily.groupby(keys, dropna=False)
    for name in STATS:
        daily['h2h_prior_' + name] = (
            grouped[name].cumsum() - daily[name]
        ).astype('float32')
    n = daily['h2h_prior_pa_count']
    for kind, base in PRIOR_RATES:
        daily['h2h_' + kind + '_rate'] = (
            daily['h2h_prior_is_' + kind] + 20 * base
        ) / (n + 20)
    features = ['h2h_prior_pa_count'] + [
        'h2h_' + kind + '_rate' for kind, _ in PRIOR_RATES
    ]
    return df.merge(
        daily[keys + ['日期'] + features],
        on=keys + ['日期'], how='left', validate='many_to_one'
    )


def prepare_data(pa_file, profile_file):
    pa_file = Path(pa_file)
    profile_file = Path(profile_file)
    if not pa_file.exists():
        raise FileNotFoundError(f'請將逐打席資料放到程式旁：{pa_file.name}')
    if profile_file.exists():
        profiles = pd.ExcelFile(profile_file)
        print('找到球員數據檔（僅檢查，不用全年資料回填歷史）：', profiles.sheet_names)
    else:
        print('未找到球員年度統計檔，仍可使用逐打席歷史特徵進行訓練。')

    df = pd.read_excel(pa_file, sheet_name='逐打席', dtype={'比賽識別碼': 'string'})
    needed = [
        '比賽日期', '比賽識別碼', '全場第幾打席', '打者', '投手', '打擊結果',
        '局數', '上下半局', '進攻球隊', '防守球隊', '打席前出局數',
        '打席前一壘有人', '打席前二壘有人', '打席前三壘有人',
        '打席前客隊分', '打席前主隊分'
    ]
    missing = [name for name in needed if name not in df.columns]
    if missing:
        raise ValueError('Excel 缺少欄位：' + str(missing))

    df = df.dropna(subset=['打擊結果', '比賽日期', '比賽識別碼']).copy()
    df = df[df['打擊結果'].astype(str).str.strip().ne('')].copy()
    df['日期'] = pd.to_datetime(df['比賽日期'], errors='coerce').dt.normalize()
    df = df.dropna(subset=['日期']).copy()
    df = df.drop_duplicates(subset=['比賽識別碼', '全場第幾打席']).copy()

    reasons = df['打擊結果'].map(excluded_reason)
    ignored_errors = int(reasons.eq('error').sum())
    ignored_sacrifices_bunts = int(reasons.eq('sacrifice_or_bunt').sum())
    df = df[reasons.isna()].copy()
    df['標籤'] = df['打擊結果'].map(outcome)
    ignored_other = int(df['標籤'].isna().sum())
    df = df[df['標籤'].isin(LABELS)].copy()
    print(
        f'已排除失誤／失策 {ignored_errors} 筆、'
        f'犧牲飛球／短打／觸擊 {ignored_sacrifices_bunts} 筆、'
        f'其他非四類結果 {ignored_other} 筆。'
    )
    if df.empty:
        raise ValueError('篩選後沒有符合出局、一安、長打、保送的逐打席資料。')

    labels = LABELS.copy()
    label_to_id = {label: i for i, label in enumerate(labels)}
    df['label_id'] = df['標籤'].map(label_to_id).astype('int32')
    for kind, label in [('out', '出局'), ('single', '一安'), ('xbh', '長打'), ('walk', '保送')]:
        df['is_' + kind] = (df['標籤'] == label).astype('int32')
    df['pa_count'] = 1

    for col in ['打者', '投手', '上下半局', '進攻球隊', '防守球隊']:
        df[col] = df[col].fillna('未知').astype(str).str.strip()

    df = prior_stats(df, '打者', 'b_')
    df = prior_stats(df, '投手', 'p_')
    df = matchup_prior_stats(df)

    cats = ['打者', '投手', '上下半局', '進攻球隊', '防守球隊']
    nums = [
        '局數', '打席前出局數', '打席前一壘有人', '打席前二壘有人',
        '打席前三壘有人', '打席前客隊分', '打席前主隊分',
        'b_prior_pa_count', 'b_out_rate', 'b_single_rate', 'b_xbh_rate', 'b_walk_rate',
        'p_prior_pa_count', 'p_out_rate', 'p_single_rate', 'p_xbh_rate', 'p_walk_rate',
        'h2h_prior_pa_count', 'h2h_out_rate', 'h2h_single_rate', 'h2h_xbh_rate', 'h2h_walk_rate'
    ]
    for col in nums:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    dates = sorted(df['日期'].unique())
    if len(dates) < 10:
        raise ValueError('篩選後不同比賽日期不足 10 天，無法切分資料。')
    cut_a, cut_b = int(len(dates) * 0.70), int(len(dates) * 0.85)
    train = df[df['日期'].isin(dates[:cut_a])].copy()
    valid = df[df['日期'].isin(dates[cut_a:cut_b])].copy()
    test = df[df['日期'].isin(dates[cut_b:])].copy()

    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), nums)
    ])
    X_train = prep.fit_transform(train[cats + nums]).astype('float32')
    X_valid = prep.transform(valid[cats + nums]).astype('float32')
    X_test = prep.transform(test[cats + nums]).astype('float32')
    y_train = train['label_id'].to_numpy()
    y_valid = valid['label_id'].to_numpy()
    y_test = test['label_id'].to_numpy()

    print(f'資料：訓練 {len(train)}、驗證 {len(valid)}、測試 {len(test)}，輸入特徵 {X_train.shape[1]}')
    print('訓練集分類筆數：', train['標籤'].value_counts().reindex(labels, fill_value=0).to_dict())
    print('驗證集分類筆數：', valid['標籤'].value_counts().reindex(labels, fill_value=0).to_dict())
    print('測試集分類筆數：', test['標籤'].value_counts().reindex(labels, fill_value=0).to_dict())

    return {
        'train': train, 'valid': valid, 'test': test,
        'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
        'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
        'prep': prep, 'cats': cats, 'nums': nums, 'labels': labels,
        'ignored_errors': ignored_errors,
        'ignored_sacrifices_bunts': ignored_sacrifices_bunts,
        'ignored_other': ignored_other
    }
