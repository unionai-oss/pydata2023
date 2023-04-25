import argparse
import os
import re
import tempfile
import typing
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import uuid
from pyspark.sql import DataFrame as sparkDF
from sklearn.datasets import make_regression
from xgboost import XGBRegressor

from tlm.helpers import (
    FEATURES,
    bar_plot_altair_html,
)


def get_s3_client():
    s3_client = boto3.client("s3", aws_access_key_id="minio",
                             aws_secret_access_key="miniostorage", endpoint_url="http://localhost:30002")
    return s3_client


def upload_file_to_s3(file_path: str, remote_bucket: str, prefix_destination: str) -> str:
    c = get_s3_client()
    c.upload_file(file_path, remote_bucket, prefix_destination)
    return f"s3://{remote_bucket}/{prefix_destination}"


def upload_dataframe(df: pd.DataFrame):
    fd, path = tempfile.mkstemp()
    df.to_parquet(path)
    upload_file_to_s3(path)


def download_files(s3_client, bucket_name, prefix, local_path):
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    ll = s3_client.list_objects(Bucket=bucket_name,
        Prefix=prefix,
    )

    prefix_parts = [x for x in prefix.split("/") if x]

    all_keys = [x["Key"] for x in ll["Contents"]]
    for k in all_keys:
        folders = k.split("/")
        new_file = local_path.joinpath(os.path.join(*folders[len(prefix_parts):]))
        new_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {bucket_name}/{k} to {str(new_file)}")
        s3_client.download_file(bucket_name, k, str(new_file))


def analyze_spark(sdf: sparkDF):
    df = sdf.toPandas()
    ...


def analyze_with_spark(df: pd.DataFrame):
    from pyspark.sql import SparkSession
    session = SparkSession.builder \
        .master("local[1]") \
        .appName("pydatademo") \
        .getOrCreate()
    sdf = session.createDataFrame(df)
    ...


def download_file_from_s3(s3_path):
    # Extract the bucket and key from the S3 path
    match = re.match(r's3://([^/]+)/(.+)', s3_path)
    if not match:
        raise ValueError(f"Invalid S3 path: {s3_path}")

    bucket, key = match.groups()

    # Create a boto3 client for S3
    s3_client = boto3.client('s3')

    # Download the file
    tmp = tempfile.mktemp()
    s3_client.download_file(bucket, key, tmp)
    return tmp


def to_html(self, html_elements: typing.List[str]) -> str:
    grid_items = "\n".join([f'<div class="grid-item">{element}</div>' for element in html_elements])
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                grid-gap: 10px;
                padding: 10px;
            }}
            .grid-item {{
                padding: 10px;
                background-color: #f1f1f1;
                border: 1px solid #ccc;
                border-radius: 5px;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: auto;
                max-height: 300px; /* Adjust this value to set the maximum height of the grid item */
            }}
            .grid-item img {{
                display: block;
                max-width: 100%;
                max-height: 100%;
            }}
        </style>
    </head>
    <body>
        <div class="grid-container">
            {grid_items}
        </div>
    </body>
    </html>
    '''



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
    # time.sleep(15)
    return df


def model_training_xgboost(
        df: pd.DataFrame, n_estimators: int, n_jobs: int, max_depth: int
):
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
    # Write to


# Scenario: Have a local parquet file. Want to upload the file and train an xg regressor
# Then save the model to another file in S3.
def run_and_save_model(parquet_file: str):
    # Get s3 client
    # Upload the file
    # Download the file
    # Extract out the df
    ...


parser = argparse.ArgumentParser(
    prog='RunXGB',
    description='download parquet file and run xgb/or upload')
parser.add_argument("parquet_location")
parser.add_argument('-n', '--estimators')  # option that takes an integer number of estimators
parser.add_argument('-j', '--jobs')  # number of jobs
parser.add_argument('-d', '--max_depth')  # depth to run on regressor


def upload_local(parquet: Path) -> str:
    prefix = uuid.uuid4().hex[:10] + f"/{parquet.name}"
    return upload_file_to_s3(str(parquet), "my-s3-bucket", prefix)


if __name__ == "__main__":
    """
    To run on server, run with:
    python boilerplate/manual_xgb.py s3://blah -n 100 -j "-1" -d 6
    To upload local file and return s3 location
    python boilerplate/manual_xgb.py /path/to/local/parquet
    """
    args = parser.parse_args()
    parquet = str(args.parquet_location)
    if not parquet.startswith("s3"):
        # Assume this is a local file - upload it to s3
        upload_local(Path(parquet))
        exit(0)

    # Else assume we're operating in training mode and run the training.
    estimators = int(args.estimators)
    jobs = int(args.jobs)
    depth = int(args.max_depth)

    print(f"Running on {parquet} with {estimators} {jobs} {depth}")
    # run_training_steps()
