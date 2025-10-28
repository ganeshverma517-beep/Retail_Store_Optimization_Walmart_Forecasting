import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True
sns.set(style="whitegrid")

#############################################
# 1. Data Load and Preparation
#############################################

# Kaggle competition mount paths
TRAIN_PATH    = "/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip"
STORES_PATH   = "/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv"
FEATURES_PATH = "/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip"

train_df    = pd.read_csv(TRAIN_PATH)
stores_df   = pd.read_csv(STORES_PATH)
features_df = pd.read_csv(FEATURES_PATH)

# Normalize column names
train_df.columns    = [c.lower() for c in train_df.columns]
stores_df.columns   = [c.lower() for c in stores_df.columns]
features_df.columns = [c.lower() for c in features_df.columns]

# Parse dates
train_df["date"]    = pd.to_datetime(train_df["date"])
features_df["date"] = pd.to_datetime(features_df["date"])

# Merge: sales + store metadata + external features
df = (
    train_df
    .merge(stores_df,   on="store", how="left")
    .merge(features_df, on=["store", "date", "isholiday"], how="left")
    .sort_values(["store", "dept", "date"])
    .reset_index(drop=True)
)

# Time helper columns
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week"]  = df["date"].dt.isocalendar().week.astype(int)


#############################################
# 2. Exploratory Data Analysis (EDA)
#############################################

def plot_total_sales_over_time(data: pd.DataFrame):
    sales_over_time = (
        data.groupby("date", as_index=False)["weekly_sales"]
            .sum()
            .rename(columns={"weekly_sales": "total_weekly_sales"})
    )

    plt.figure()
    plt.plot(sales_over_time["date"], sales_over_time["total_weekly_sales"])
    plt.title("Total Weekly Sales Over Time (All Stores)")
    plt.xlabel("Date")
    plt.ylabel("Total Weekly Sales")
    plt.tight_layout()
    plt.show()

    return sales_over_time


def plot_holiday_vs_nonholiday(data: pd.DataFrame):
    holiday_perf = (
        data.groupby("isholiday")["weekly_sales"]
            .mean()
            .reset_index()
            .rename(columns={"weekly_sales": "avg_sales"})
    )

    plt.figure()
    sns.barplot(data=holiday_perf, x="isholiday", y="avg_sales")
    plt.title("Holiday vs Non-Holiday Avg Weekly Sales")
    plt.xlabel("IsHoliday")
    plt.ylabel("Avg Weekly Sales")
    plt.tight_layout()
    plt.show()

    return holiday_perf


def plot_top_bottom_stores(data: pd.DataFrame, top_n: int = 10):
    store_perf = (
        data.groupby("store", as_index=False)["weekly_sales"]
            .mean()
            .rename(columns={"weekly_sales": "avg_weekly_sales"})
            .sort_values(by="avg_weekly_sales", ascending=False)
    )

    top_stores = store_perf.head(top_n)
    bottom_stores = store_perf.tail(top_n)

    plt.figure()
    sns.barplot(data=top_stores, x="store", y="avg_weekly_sales")
    plt.title(f"Top {top_n} Stores by Avg Weekly Sales")
    plt.xlabel("Store")
    plt.ylabel("Avg Weekly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.barplot(data=bottom_stores, x="store", y="avg_weekly_sales")
    plt.title(f"Bottom {top_n} Stores by Avg Weekly Sales")
    plt.xlabel("Store")
    plt.ylabel("Avg Weekly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return store_perf


def plot_seasonality_heatmap(data: pd.DataFrame):
    seasonality = (
        data.groupby(["store", "month"], as_index=False)["weekly_sales"]
            .mean()
            .rename(columns={"weekly_sales": "avg_sales_month"})
    )

    pivot_tbl = seasonality.pivot(
        index="store",
        columns="month",
        values="avg_sales_month"
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_tbl, cmap="viridis")
    plt.title("Store vs Month: Avg Sales Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Store")
    plt.tight_layout()
    plt.show()

    return pivot_tbl


# Run EDA steps
sales_over_time_tbl = plot_total_sales_over_time(df)
holiday_perf_tbl    = plot_holiday_vs_nonholiday(df)
store_perf_tbl      = plot_top_bottom_stores(df, top_n=10)
seasonality_tbl     = plot_seasonality_heatmap(df)


#############################################
# 3. Store Feature Engineering for Clustering
#############################################

# Average performance per store
feat = df.groupby("store").agg(
    avg_weekly_sales = ("weekly_sales", "mean"),
    sales_volatility = ("weekly_sales", "std"),
    store_type       = ("type", "first"),
    store_size       = ("size", "first"),
).reset_index()

# Holiday uplift % = relative lift of holiday weeks vs non-holiday
holiday_stats = (
    df.groupby(["store", "isholiday"])["weekly_sales"]
        .mean()
        .reset_index()
        .pivot(index="store", columns="isholiday", values="weekly_sales")
        .rename(columns={False: "nonholiday_avg", True: "holiday_avg"})
        .fillna(0)
)

holiday_stats["holiday_uplift_pct"] = np.where(
    holiday_stats["nonholiday_avg"] > 0,
    (holiday_stats["holiday_avg"] - holiday_stats["nonholiday_avg"])
    / holiday_stats["nonholiday_avg"],
    0,
)

holiday_stats = holiday_stats[["holiday_uplift_pct"]].reset_index()
feat = feat.merge(holiday_stats, on="store", how="left")

# Trend slope (linear regression vs time index, per store)
slopes = []
for store_id, g in (
    df[["store", "date", "weekly_sales"]]
        .sort_values(["store", "date"])
        .groupby("store")
):
    t = np.arange(len(g)).reshape(-1, 1)
    y = g["weekly_sales"].values

    if len(g) > 1 and np.std(y) > 0:
        lr = LinearRegression()
        lr.fit(t, y)
        slope = lr.coef_[0]
    else:
        slope = 0.0

    slopes.append({"store": store_id, "trend_slope": slope})

slopes = pd.DataFrame(slopes)
feat = feat.merge(slopes, on="store", how="left")


#############################################
# 4. Clustering (K-Means)
#############################################

cluster_features = [
    "avg_weekly_sales",
    "sales_volatility",
    "holiday_uplift_pct",
    "trend_slope",
    "store_size",
]

X = feat[cluster_features].fillna(0).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
feat["cluster"] = kmeans.fit_predict(X_scaled)

cluster_profile = (
    feat.groupby("cluster")[cluster_features]
        .mean()
        .round(2)
        .reset_index()
)

plt.figure()
sns.barplot(
    data=feat,
    x="cluster",
    y="avg_weekly_sales",
    estimator=np.mean,
    errorbar=None
)
plt.title("Average Weekly Sales by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Avg Weekly Sales")
plt.tight_layout()
plt.show()

# Save clustering output
feat.to_csv("/kaggle/working/Store_Clusters.csv", index=False)


#############################################
# 5. Forecasting (Moving Average + ARIMA)
#############################################

# choose one store to forecast (first store in df)
target_store = df["store"].iloc[0]

store_ts = (
    df[df["store"] == target_store]
    .groupby("date", as_index=False)["weekly_sales"]
    .sum()
    .sort_values("date")
    .reset_index(drop=True)
)

store_ts = store_ts.set_index("date")
store_ts["moving_avg_4w"] = store_ts["weekly_sales"].rolling(window=4).mean()

plt.figure()
plt.plot(store_ts.index, store_ts["weekly_sales"], label="Actual")
plt.plot(store_ts.index, store_ts["moving_avg_4w"], label="4-Week Moving Avg", linewidth=3)
plt.title(f"Store {target_store} Weekly Sales vs 4-Week Moving Average")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()
plt.show()

# ARIMA(1,1,1)
ts_values = store_ts["weekly_sales"].astype(float)

model = ARIMA(ts_values, order=(1, 1, 1))
model_fit = model.fit()

n_forecast = 8
forecast_res = model_fit.get_forecast(steps=n_forecast)
forecast_mean = forecast_res.predicted_mean
forecast_ci   = forecast_res.conf_int(alpha=0.2)

last_date = store_ts.index.max()
future_dates = pd.date_range(start=last_date, periods=n_forecast + 1, freq="W")[1:]

plt.figure()
plt.plot(store_ts.index, store_ts["weekly_sales"], label="Historical")
plt.plot(future_dates, forecast_mean, label="Forecast", linewidth=3)
plt.fill_between(
    future_dates,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.3,
    label="Forecast CI"
)
plt.title(f"Store {target_store} ARIMA Forecast (Next {n_forecast} Weeks)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()
plt.show()

forecast_table = pd.DataFrame({
    "date": future_dates,
    "forecasted_sales": forecast_mean.round(2),
    "lower_ci": forecast_ci.iloc[:, 0].round(2),
    "upper_ci": forecast_ci.iloc[:, 1].round(2),
})

forecast_table.to_csv("/kaggle/working/Store_Forecast.csv", index=False)


#############################################
# 6. Summary Output to Console
#############################################

print("\n=== DATA MERGE SHAPES ===")
print("Merged df:", df.shape)

print("\n=== HOLIDAY VS NON-HOLIDAY SALES ===")
print(holiday_perf_tbl)

print("\n=== STORE PERFORMANCE (HEAD/TAIL) ===")
print(store_perf_tbl.head())
print(store_perf_tbl.tail())

print("\n=== CLUSTER PROFILE ===")
print(cluster_profile)

print("\n=== FORECAST (NEXT 8 WEEKS) ===")
print(forecast_table)

print("\nArtifacts saved to /kaggle/working/:")
print("- Store_Clusters.csv")
print("- Store_Forecast.csv")
print("\nDone.")
