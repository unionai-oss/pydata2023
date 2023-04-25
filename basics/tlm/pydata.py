import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from flytekit import Resources, WorkflowFailurePolicy, task, workflow, Deck
from sklearn.datasets import make_regression
from xgboost import XGBRegressor

from .helpers import (
    EARLIEST_DATE,
    FEATURES,
    bar_plot_altair_html,
    load_items_df,
)


@task(
    requests=Resources(cpu="4", mem="8Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
    disable_deck=False,
)
def etl_sales_aggregation(start_dt: datetime) -> pd.DataFrame:
    """Generates fake sales data and aggregates it by year, month, id, name and calculates sum of sales.
    Mimics an ETL job.

    Args:
        start_dt (datetime): Start date for generating fake sales data
    Returns:
        pd.DataFrame: Aggregated sales data
    """
    items_df = load_items_df()

    df = pd.DataFrame()
    current_date = max(EARLIEST_DATE, start_dt)
    end_date = datetime.now().astimezone(pytz.utc)
    memory = np.zeros([1_000_000, 10], dtype=np.float32)
    # iterate days from max(earliest_date, start_dt) to present day
    i = 0
    while current_date <= end_date:
        n_sales = 1000 + int(abs(np.random.normal(loc=0, scale=2000, size=1))[0])
        item_sample = np.random.choice(items_df.id, p=items_df.weight, size=n_sales)

        sample_df = pd.merge(pd.DataFrame({"id": item_sample}), items_df, on="id")
        sample_df["date"] = current_date
        df = pd.concat([df, sample_df], axis=0)

        if i % 50 == 0:
            print(f"Iteration {i}")

        # make mem usage more realistic
        n = 1_000_000 if i % 3 == 0 else 100_000
        memory = np.concatenate(
            [memory, np.zeros([int(n * random.random()), 10], dtype=np.float32)],
            axis=0,
        )
        current_date += timedelta(days=1)
        i += 1

    # group by year, month, id, name and calculate sum of sales
    print(df.shape, len(df.id.unique()))
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly_aggs_df = (
        df.groupby(["year", "month", "id", "name"])["price"]
        .sum()
        .reset_index()
        .rename(columns={"price": "sales"})
    )

    top_sales = monthly_aggs_df.sort_values(
        by=["year", "month", "sales"], ascending=False
    ).head(25)

    Deck("Current Top Sales", top_sales.to_html())

    return monthly_aggs_df


@task(requests=Resources(cpu="2", mem="4Gi"), limits=Resources(cpu="2", mem="4Gi"))
def etl_prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a simulated training dataset. Mimics a preprocessing step

    Returns:
        pd.DataFrame: Training data
    """
    X, y, coef = make_regression(
        n_samples=1_000_000, n_features=100, n_informative=20, coef=True
    )

    idx = np.argsort(coef)[::-1]
    columns = [f"x_{i}" for i in range(X.shape[1])]

    # give columns meaningful names
    for i, sorted_i in enumerate(idx):
        if i < 20:
            columns[sorted_i] = FEATURES[i]

    df = pd.DataFrame(X, columns=columns)
    df["y"] = y
    print(df)
    return df


@task(
    requests=Resources(cpu="15", mem="5Gi"),
    limits=Resources(cpu="15", mem="5Gi"),
    disable_deck=False,
)
def model_training_xgboost(
        df: pd.DataFrame, n_estimators: int, n_jobs: int, max_depth: int
):
    """Trains a XGBoost model and displays feature importance.

    Args:
        df (pd.DataFrame): Training data
        n_estimators (int): Number of trees
        n_jobs (int): Number of jobs
        max_depth (int): Max depth of trees
    """
    model = XGBRegressor(
        n_estimators=n_estimators, n_jobs=n_jobs, verbosity=1, max_depth=max_depth
    )
    X, y = df.drop("y", axis=1), df["y"]

    model.fit(X, y)

    importance_df = pd.DataFrame(
        {"importance": model.feature_importances_, "names": X.columns}
    )
    top_importance = importance_df.sort_values(by="importance", ascending=False).head(
        20
    )

    fi_altair = bar_plot_altair_html(top_importance, "importance", "names")
    Deck("Feature Importance", fi_altair)


@workflow
def forecasting_wf(start_dt: datetime):
    """A Demo sales forecasting model training workflow. It has
    two major steps: ETL and model training. Note
        - etl_sales_aggregation is slow on memory
        - model_training_xgboost is cpu bound until 15 cpus

    Args:
        start_dt (datetime): Start date for generating fake sales data
    """
    sales_df = etl_sales_aggregation(start_dt=start_dt)
    train_df = etl_prep_features(df=sales_df)

    for cpu in [1, 4, 12, 15]:
        model_training_xgboost(
            df=train_df, n_estimators=100, n_jobs=-1, max_depth=6
        ).with_overrides(
            requests=Resources(cpu=str(cpu), mem="8Gi"),
            limits=Resources(cpu=str(cpu), mem="8Gi"),
            name=f"model_training_xgboost_cpu_{cpu}",
        )