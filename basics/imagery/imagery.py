import os
import random
import tempfile
from dataclasses import asdict, dataclass
from typing import List, Tuple

import pandas as pd
from dataclasses_json import dataclass_json
from flytekit import StructuredDataset, dynamic, kwtypes, task, workflow, Deck
from flytekit.types.file import FlyteFile, HTMLPage, PythonNotebook
from flytekitplugins.papermill import NotebookTask
from PIL import Image

from imagery.renderers import FancyGrid, ImageRenderer

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass_json
@dataclass
class IMG:
    file: FlyteFile
    description: str
    prediction: float
    label: float


@task(cache=False, disable_deck=False)
def score_image(name: str) -> IMG:
    """Returns a score and a description of the image.

    Args:
        name (str): The name of the image to score.
    Returns:
        IMG: a dataclass containing the image, a description, a prediction, and a label.
    """
    path = os.path.join(CURR_DIR, name)
    img = Image.open(path)
    Deck("Image", ImageRenderer().to_html(img))

    prediction = 0.85 + random.random() * 0.14
    if "img3" in name:
        description = f"This image is named {name} and contains clouds"
        label = 0
    else:
        description = f"This image is named {name} and contains hydrocarbon plumes"
        label = 1

    return IMG(
        file=FlyteFile(path=path),
        description=description,
        label=label,
        prediction=prediction,
    )


@task(disable_deck=False)
def display_grid(images: List[IMG]):
    """Displays a grid of images"""
    files = [ImageRenderer().to_html(img.file) for img in images]
    Deck("Grid", FancyGrid().to_html(files))


@task
def images_to_df(images: List[IMG]) -> FlyteFile:
    """Converts a list of images to a dataframe and saves it to a parquet file"""
    df = pd.DataFrame(
        [{**asdict(img), "remote_source": img.file.remote_source} for img in images]
    )
    tmp = tempfile.mktemp()
    df.to_parquet(tmp)
    return FlyteFile(path=tmp)


@task
def get_remote_source(ff: FlyteFile) -> str:
    """Returns the remote source of a FlyteFile"""
    return ff.remote_source


@workflow
def report_preprocessing(images: List[IMG]) -> str:
    """A workflow that preprocesses images for the quality report. This
    could be simplified from flytekit"""
    ff = images_to_df(images=images)
    source = get_remote_source(ff=ff)
    return source


quality_report = NotebookTask(
    name="quality_report",
    notebook_path=os.path.join(CURR_DIR, "demo_display.ipynb"),
    render_deck=True,
    disable_deck=False,
    inputs=kwtypes(path=str),
    outputs=kwtypes(out_nb=PythonNotebook, out_rendered_nb=HTMLPage),
)


@workflow
def wf() -> Tuple[PythonNotebook, HTMLPage]:
    """A demo workflow that "scores images" and displays them
    in Flyte Decks and in a Jupyter Notebooks.
    """
    images = [
        score_image(name="img1.png"),
        score_image(name="img2.png"),
        score_image(name="img3.png"),
    ]
    display_grid(images=images)

    path = report_preprocessing(images=images)
    out, render = quality_report(path=path)
    return out, render
