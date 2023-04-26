import pandas as pd

from sklearn.linear_model import LogisticRegression
from flytekit import task, dynamic, workflow

import base64
from io import BytesIO, StringIO
from typing import List, Union, Dict, Tuple
import numpy as np
from flytekit.types.file import FlyteFile
from PIL import Image
import flytekit

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import altair as alt
from sklearn.datasets import make_classification


class ImageRenderer:
    """Converts a FlyteFile or PIL.Image.Image object to an HTML string with the image data
    represented as a base64-encoded string.
    """

    def to_html(cls, image_src: Union[FlyteFile, Image.Image]) -> str:
        img = cls._get_image_object(image_src)
        return cls._image_to_html_string(img)

    @staticmethod
    def _get_image_object(image_src: Union[FlyteFile, Image.Image]) -> Image.Image:
        if isinstance(image_src, FlyteFile):
            local_path = image_src.download()
            return Image.open(local_path)
        elif isinstance(image_src, Image.Image):
            return image_src
        else:
            raise ValueError("Unsupported image source type")

    @staticmethod
    def _image_to_html_string(img: Image.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'


@task(container_image="{{.images.xgb.fqn}}:{{.images.xgb.version}}", cache=True, cache_version="1.0")
def load_data() -> pd.DataFrame:
    """Get the dataset."""
    n_features = 40
    X, y = make_classification(n_samples=25_000, n_features=n_features,
                               n_informative=15,
                               n_repeated=10,
                               n_redundant=15
                               )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df


@task(container_image="{{.images.xgb.fqn}}:{{.images.xgb.version}}")
def etl_preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A place holder that would exist IRL"""
    data["engineered"] = data["feature_1"]**2 + data["feature_2"]

    msk = np.random.rand(len(data)) < 0.8

    return data[msk], data[~msk]


@task(container_image="{{.images.xgb.fqn}}:{{.images.xgb.version}}")
def train_model(data: pd.DataFrame, hyperparameters: dict) -> LogisticRegression:
    """Train a model on the wine dataset."""
    features = data.drop("target", axis="columns")
    target = data["target"]
    return LogisticRegression(**hyperparameters).fit(features, target)


def plot_roc_auc(roc_df) -> str:
    roc_line = alt.Chart(roc_df).mark_line(color = 'purple').encode(
                                                                alt.X('fpr', title="false positive rate"),
                                                                alt.Y('tpr', title="true positive rate"))

    roc = alt.Chart(roc_df).mark_area(fillOpacity = 0.25, fill = 'purple').encode(
                                                                    alt.X('fpr', title="false positive rate"),
                                                                    alt.Y('tpr', title="true positive rate"))

    chart = roc_line + roc

    str_io = StringIO()
    chart.save(str_io, format="html", embed_options={'renderer': 'svg'})
    return str_io.getvalue()


@task(container_image="{{.images.xgb.fqn}}:{{.images.xgb.version}}", disable_deck=False)
def validate_model(model: LogisticRegression, data: pd.DataFrame) -> float:
    """Validate the model on the wine dataset."""
    features = data.drop("target", axis="columns")
    target = data["target"]
    scores = model.predict_proba(features)[:, 1]

    # Compute the ROC AUC curve
    roc_auc_score(target, scores)
    # plot ROC curve

    fpr, tpr, thresholds = roc_curve(target, scores)
    plt.plot(fpr, tpr)
    
    roc_html = plot_roc_auc(roc_df=pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds}))
    flytekit.Deck("ROC Curve", roc_html)

    score = model.score(features, target)

    return float(score)


@task(container_image="{{.images.xgb.fqn}}:{{.images.xgb.version}}", disable_deck=False)
def performance_report(results: Dict[str, float]):
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    table = pd.DataFrame(sorted_results, columns=["Model Name", "Accuracy"])

    flytekit.Deck("Results", table.to_html())


@dynamic(container_image="{{.images.xgb.fqn}}:{{.images.xgb.version}}")
def training_workflow(regularization: List[float]):
    """Put all of the steps together into a single workflow."""
    data = load_data()
    train, test = etl_preprocess_data(data=data)

    results = {}
    for c in regularization:
        if c < 1:
            cn = "{:.0E}".format(c)
        else:
            cn = int(c)
        name = f"train_model[C={str(cn)}]"
        model = train_model(
            data=train,
            hyperparameters={"C": c},
        ).with_overrides(name=f"train_model[C={str(cn)}]")
        acc = validate_model(model=model, data=test).with_overrides(
            name=f"validate_model[C={str(cn)}]")
        results[name] = acc

    performance_report(results=results)


@workflow
def run_training():
    training_workflow(regularization=[0.001, 10.0])
