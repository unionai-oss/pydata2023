





commands
setup 
aws sso login --profile uai
docker buildx build --file Dockerfile --build-arg VERSION=1.5.0 --platform linux/arm64,linux/amd64 --tag ghcr.io/flyteorg/flytecookbook:pydata_img_1 --push .


run time
cd basics
pyflyte -c ~/.flyte/ucdemo.yaml register imagery/imagery.py
