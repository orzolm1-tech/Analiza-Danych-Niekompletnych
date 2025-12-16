import numpy as np
import pandas as pd

import missingno as msno
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

#Wczytanie danych
data = Path(r"C:\Users\micha\PycharmProjects\PythonProject2\dane_niekompletne")

train_path  = data / "pzn-rent-train.csv"
test_path   = data / "pzn-rent-test.csv"
sample_path = data / "pzn-sample-sub.csv"

df_train  = pd.read_csv(train_path)
df_test   = pd.read_csv(test_path)
df_sample = pd.read_csv(sample_path)

print("TRAIN shape:", df_train.shape)
print("TEST  shape:", df_test.shape)
print("SAMPLE shape:", df_sample.shape)
print(df_train.head())

#Raport sentyneli
def report_sentinel_minus999(df, name):
    mask = (df == -999)
    cols_with_999 = mask.any()
    cols = cols_with_999[cols_with_999].index.tolist()
    print(f"=== {name}: kolumny z -999 ===")
    for col in cols:
        cnt = mask[col].sum()
        print(f"{col}: {cnt} wystąpień")
    if not cols:
        print("brak -999")
    print()

report_sentinel_minus999(df_train, "TRAIN")
report_sentinel_minus999(df_test,  "TEST")

# Podstawowa tabela braków
na_cnt = df_train.isna().sum().sort_values(ascending=False)
na_pct = (df_train.isna().mean() * 100).sort_values(ascending=False).round(2)
missing_summary = pd.DataFrame({"n_missing": na_cnt, "pct_missing": na_pct})
print("=== Missing summary (top 10) ===")
print(missing_summary.head(10))

# Wizualizacja braków
msno.bar(df_train)
plt.show()
msno.matrix(df_train)
plt.show()
msno.heatmap(df_train)
plt.show()

# Konwersja dat + sanity check na datach
date_cols = ["date_activ", "date_modif", "date_expire"]

for df in [df_train, df_test]:
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

print("=== Typy dat po konwersji (TRAIN) ===")
print(df_train[date_cols].dtypes)

# kolejnosc dat
cond_bad_order = (
    (df_train["date_activ"] > df_train["date_modif"]) |
    (df_train["date_modif"] > df_train["date_expire"])
)
df_bad_dates = df_train[cond_bad_order]
print("Złe kolejności dat (TRAIN):", df_bad_dates.shape[0])

# czas trwania ogłoszenia + anomalie
for df_name, df in [("train", df_train), ("test", df_test)]:
    duration = (df["date_expire"] - df["date_activ"]).dt.days

    df["listing_duration"] = duration
    df["duration_anomaly"] = (duration >= 365).astype(int)
    df.loc[duration >= 365, "listing_duration"] = np.nan

    print(f"=== {df_name} listing_duration ===")
    print(duration.describe())
    print("<=0 dni:", (duration <= 0).sum())
    print(">=365 dni:", (duration >= 365).sum())
    print()

#Sanity check: price / flat_rent / flat_deposit
for df_name, df in [("train", df_train), ("test", df_test)]:
    print(f"=== {df_name} price & rent sanity ===")
    print("price <= 0:", (df["price"] <= 0).sum() if "price" in df.columns else "brak")
    print("flat_rent < 0:", (df["flat_rent"] < 0).sum())
    print("flat_deposit < 0:", (df["flat_deposit"] < 0).sum())
    print()

# anomalia: flat_rent > price
mask_anomaly_rent = df_train["flat_rent"] > df_train["price"]
print("Liczba anomalii price/rent (flat_rent > price):", mask_anomaly_rent.sum())

df_train["rent_over_price_anomaly"] = mask_anomaly_rent.astype(int)

df_train.loc[mask_anomaly_rent, "flat_rent"] = np.nan
df_test["rent_over_price_anomaly"] = 0

# checkpoint
print("Po czyszczeniu rent vs price – ile NaN w flat_rent (TRAIN):",
      df_train["flat_rent"].isna().sum())

#Cleaning flat_area
for df_name, df in [("train", df_train), ("test", df_test)]:
    print(f"=== {df_name} flat_area przed czyszczeniem ===")
    print(df["flat_area"].describe())
    print("flat_area <= 5:", (df["flat_area"] <= 5).sum())
    print("flat_area > 200:", (df["flat_area"] > 200).sum())
    print()

def clean_flat_area(df):
    area = df["flat_area"]

    df["flat_area_sentinel"]   = (area == -999).astype(int)
    df["flat_area_too_small"]  = (area <= 5).astype(int)
    df["flat_area_too_big"]    = (area > 200).astype(int)

    mask_anomaly = (area == -999) | (area <= 5) | (area > 200)
    df.loc[mask_anomaly, "flat_area"] = np.nan
    return df

df_train = clean_flat_area(df_train)
df_test  = clean_flat_area(df_test)

for df_name, df in [("train", df_train), ("test", df_test)]:
    print(f"=== {df_name} flat_area po czyszczeniu ===")
    print(df["flat_area"].describe())
    print("NaN w flat_area:", df["flat_area"].isna().sum())
    print()

#Cleaning flat_rooms + area_per_room
for df_name, df in [("train", df_train), ("test", df_test)]:
    print(f"=== {df_name} flat_rooms przed czyszczeniem ===")
    print(df["flat_rooms"].describe())
    print("flat_rooms <= 0:", (df["flat_rooms"] <= 0).sum())
    print("flat_rooms > 10:", (df["flat_rooms"] > 10).sum())
    print()

def clean_flat_rooms(df):
    rooms = df["flat_rooms"]
    df["flat_rooms_invalid"] = (rooms <= 0).astype(int)
    df["flat_rooms_large"]   = (rooms > 10).astype(int)

    df.loc[rooms <= 0, "flat_rooms"] = np.nan
    df.loc[rooms > 10, "flat_rooms"] = np.nan
    return df

df_train = clean_flat_rooms(df_train)
df_test  = clean_flat_rooms(df_test)

# area_per_room dla TRAIN i TEST
for df in [df_train, df_test]:
    df["area_per_room"] = df["flat_area"] / df["flat_rooms"]

print("=== TRAIN area_per_room przed czyszczeniem ===")
print(df_train["area_per_room"].describe())
print("area_per_room < 5:", (df_train["area_per_room"] < 5).sum())
print("area_per_room > 80:", (df_train["area_per_room"] > 80).sum())
print()

def clean_area_per_room(df):
    apr = df["area_per_room"]
    df["area_per_room_small"] = (apr < 5).astype(int)
    df["area_per_room_large"] = (apr > 80).astype(int)
    df.loc[(apr < 5) | (apr > 80), "area_per_room"] = np.nan
    return df

df_train = clean_area_per_room(df_train)
df_test  = clean_area_per_room(df_test)

print("=== TRAIN area_per_room po czyszczeniu ===")
print(df_train["area_per_room"].describe())
print("NaN w area_per_room:", df_train["area_per_room"].isna().sum())
print()

#Furnished inconsistency (niemeblowane + sprzęty)
def fix_furnished_inconsistency(df):
    equip_cols = [
        "flat_fridge", "flat_cooker", "flat_oven",
        "flat_washmachine", "flat_dishwasher", "flat_television"
    ]
    has_equipment = df[equip_cols].sum(axis=1) > 0

    mask_incons = (df["flat_furnished"] == False) & has_equipment
    df["furnished_inconsistency"] = mask_incons.astype(int)
    df.loc[mask_incons, "flat_furnished"] = np.nan
    return df

df_train = fix_furnished_inconsistency(df_train)
df_test  = fix_furnished_inconsistency(df_test)

print("Nieumeblowane, ale sprzęty (TRAIN):",
      df_train["furnished_inconsistency"].sum())
print("NaN w flat_furnished (TRAIN):",
      df_train["flat_furnished"].isna().sum())
print()

#Cleaning quarter (bez słownika, rare -> other)
print(df_train["quarter"].value_counts(dropna=False).head(15))
print("missing quarter train:", df_train["quarter"].isna().sum())
print("missing quarter test:", df_test["quarter"].isna().sum())

for df in [df_train, df_test]:
    df["quarter_missing"] = df["quarter"].isna().astype(int)
    q = (
        df["quarter"]
        .astype(str)
        .str.lower()
        .str.strip()
    )
    q = q.replace("nan", np.nan)
    df["quarter"] = q

# rare -> other (na podstawie TRAIN)
counts = df_train["quarter"].value_counts(dropna=True)
rare = counts[counts < 20].index

for df in [df_train, df_test]:
    df["quarter_clean"] = df["quarter"].where(~df["quarter"].isin(rare), other="other")
    df["quarter_clean"] = df["quarter_clean"].fillna("unknown")

print("=== quarter_clean value_counts (TRAIN, top 15) ===")
print(df_train["quarter_clean"].value_counts().head(15))
print()

#Kopia do feature engineering
train_fe = df_train.copy()
test_fe  = df_test.copy()

#Cechy z dat
for df in [train_fe, test_fe]:
    df["time_to_modif"] = (df["date_modif"] - df["date_activ"]).dt.days
    df["activ_year"]    = df["date_activ"].dt.year
    df["activ_month"]   = df["date_activ"].dt.month
    df["activ_dow"]     = df["date_activ"].dt.dayofweek

#Cechy z tytułu ogłoszenia
for df in [train_fe, test_fe]:
    df["ad_title_filled"] = df["ad_title"].fillna("")
    df["ad_title_len"]    = df["ad_title_filled"].str.len()
    df["ad_title_words"]  = df["ad_title_filled"].str.split().str.len()

#Markery braków dla kluczowych numeryków
marker_cols = ["flat_area", "flat_rooms", "flat_rent", "flat_deposit", "building_floor_num"]
for df in [train_fe, test_fe]:
    for col in marker_cols:
        df[col + "_missing"] = df[col].isna().astype(int)

#ilość NaN po FE
print("NaN summary (TRAIN_FE, top 15):")
print(train_fe.isna().sum().sort_values(ascending=False).head(15))
print()

#Przygotowanie danych do modelu
TARGET = "price"
ID_COL = "id"

y = train_fe[TARGET]
X = train_fe.drop(columns=[TARGET])

# kolumny do dropnięcia
cols_to_drop = date_cols + ["ad_title", "ad_title_filled", "quarter"]
X = X.drop(columns=cols_to_drop)
test_model = test_fe.drop(columns=cols_to_drop)

# num_cols = wszystkie float/int poza ID
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
num_cols = [c for c in num_cols if c not in [ID_COL]]

cat_cols = X.columns.difference(num_cols + [ID_COL]).tolist()

print("Liczba zmiennych numerycznych:", len(num_cols))
print("Liczba zmiennych kategorycznych:", len(cat_cols))

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

#Preprocessing: imputacja + kodowanie
numeric_imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(
        n_estimators=50,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    max_iter=10,
    initial_strategy="median",
    random_state=RANDOM_STATE
)

numeric_transformer = Pipeline(steps=[
    ("imputer", numeric_imputer)
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

#Model bazowy: Linear Regression
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

model_lin = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

model_lin.fit(X_train, y_train_log)

y_pred_log_lin = model_lin.predict(X_valid)
y_pred_lin = np.expm1(y_pred_log_lin)

mse_lin  = mean_squared_error(y_valid, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
mae_lin  = mean_absolute_error(y_valid, y_pred_lin)

print(f"Linear Regression: RMSE={rmse_lin:.2f}, MAE={mae_lin:.2f}")

#XGBoost – tuning + early stopping (log-target)
y_full_log = np.log1p(y)

# dopasowanie preprocessora na całym X
preprocessor.fit(X)

X_train_t = preprocessor.transform(X_train)
X_valid_t = preprocessor.transform(X_valid)
X_full_t  = preprocessor.transform(X)
X_test_t  = preprocessor.transform(test_model)

xgb_base = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    tree_method="hist",
)

param_distributions = {
    "n_estimators":     [400, 600, 800, 1000],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "max_depth":        [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 7],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda":       [0.1, 1.0, 10.0],
    "reg_alpha":        [0.0, 0.1, 1.0],
    "gamma":            [0.0, 0.1, 1.0],
}

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=40,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

random_search.fit(X_train_t, y_train_log)

print("Najlepsze parametry (log-target):")
print(random_search.best_params_)

best_cv_mse  = -random_search.best_score_
best_cv_rmse = np.sqrt(best_cv_mse)
print(f"CV RMSE na skali log(price): {best_cv_rmse:.4f}")

best_xgb_no_es = random_search.best_estimator_

y_valid_pred_log = best_xgb_no_es.predict(X_valid_t)
y_valid_pred     = np.expm1(y_valid_pred_log)

mse_valid  = mean_squared_error(y_valid, y_valid_pred)
rmse_valid = np.sqrt(mse_valid)
mae_valid  = mean_absolute_error(y_valid, y_valid_pred)

print(f"XGB log-target (bez ES) – hold-out: RMSE={rmse_valid:.2f}, MAE={mae_valid:.2f}")

#XGBoost z early stopping
best_params = random_search.best_params_.copy()

xgb_es = XGBRegressor(
    **best_params,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=50
)

xgb_es.fit(
    X_train_t, y_train_log,
    eval_set=[(X_valid_t, y_valid_log)],
    verbose=False
)

print("Best iteration (ES):", xgb_es.best_iteration)

y_pred_log = xgb_es.predict(X_valid_t)
y_pred = np.expm1(y_pred_log)

mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_valid, y_pred)

print("XGB + ES – RMSE:", rmse)
print("XGB + ES – MAE:", mae)

#Finalny model na pełnym train + predykcja na test
n_best = xgb_es.best_iteration

best_params_clean = best_params.copy()
best_params_clean.pop("n_estimators", None)

xgb_full = XGBRegressor(
    **best_params_clean,
    n_estimators=n_best,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="rmse"
)

xgb_full.fit(X_full_t, y_full_log)

y_pred_log_test = xgb_full.predict(X_test_t)
y_pred_test = np.expm1(y_pred_log_test)

submission_log_es = pd.DataFrame({
    "ID": test_model[ID_COL],
    "TARGET": y_pred_test
})

submission_log_es.to_csv("submission_xgb_log_es.csv", index=False)
print("Zapisano submission_xgb_log_es.csv")

print(submission_log_es.shape)
print(submission_log_es.head())
print(submission_log_es.isna().sum())
