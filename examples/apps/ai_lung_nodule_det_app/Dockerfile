FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y software-properties-common gcc git cmake

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-aarch64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR}
ENV PATH=/opt/conda/bin:${PATH}

RUN conda create -n cucim -c rapidsai -c conda-forge cucim
SHELL ["conda", "run", "-n", "cucim", "/bin/bash", "-c"]

RUN cd /tmp & \
    git clone --recurse-submodules https://github.com/pydicom/pylibjpeg-libjpeg && \
    cd pylibjpeg-libjpeg && \
    pip install . && \
    pip install highdicom nibabel

WORKDIR /workdir
COPY . .

RUN pip install -r requirements.txt && \
    pip install monai[all]==0.9.0 monai-deploy-app-sdk==0.4.0 && \
    rm requirements.txt

ENTRYPOINT ["conda", "run", "-n", "cucim", \
            "monai-deploy", "exec", "--input", "/input", "--output", "/output", "-l", "DEBUG", "--model", "/workdir/model/model.ts", "--workdir", "/workdir", "."]
