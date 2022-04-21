FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt

RUN pushd /tmp & \
    git clone --recurse-submodules https://github.com/pydicom/pylibjpeg-libjpeg && \
    pushd pylibjpeg-libjpeg && \
    pip install . && \
    pip install highdicom nibabel && \
    popd && \
    popd

WORKDIR /workdir
COPY . .

RUN pip install -r requirements.txt && \
    pip install monai==0.9.0 monai-deploy-app-sdk==0.4.0 torch && \
    rm requirements.txt

ENTRYPOINT monai-deploy exec --input /input --output /output -l DEBUG --model /workdir/model/model.ts --workdir /workdir .
