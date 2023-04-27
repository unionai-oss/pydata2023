# Pydata 2023 Flyte Examples

These are some simple examples that demonstrate the basic use-cases of Flyte that were written for Pydata 2023.

## Setup
Create a virtualenv and activate it. Then `pip install -r requirements.txt` as normal.

## Commands

(For the demo: `aws sso login --profile uai`)

### Register
This will compile your user code into Flyte components (tasks & workflows), and then ship them off to a Flyte backend.
When you do this, flytekit will associate a container image for those tasks that require a container image. You can build the image in a couple ways. 

#### Manual Image Building
A Dockerfile has been included in this repo. To build a multi-arch image, use a command like this.

```bash
docker buildx build --file Dockerfile --build-arg VERSION=1.5.0 --platform linux/arm64,linux/amd64 --tag ghcr.io/flyteorg/flytecookbook:pydata_2 --push .
```

Use this registration command will use the image for both examples, but you will have to modify the tasks in the `demo.py` file, remove the `container_image` argument from the `task` decorator.

```bash
pyflyte -c ~/.flyte/ucdemo.yaml register --image ghcr.io/flyteorg/flytecookbook:pydata_2 --image xgb=ghcr.io/flyteorg/flytecookbook:pydata_2 imagery cpu_compare
```

#### envd Image Building
Flytekit also has recently released a beta feature to help with the image building process. This is the `ImageSpec` object declared in the `demo.py` file. This can also be declared via a file, which is what is used for the second example. 

```bash
pyflyte -c ~/.flyte/ucdemo.yaml register --image xgb=xgb_img.yaml imagery cpu_compare lr
pyflyte -c ~/.flyte/dev.yaml register --image xgb=xgb_img.yaml imagery cpu_compare lr
```
