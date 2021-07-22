# use the base image given or any custom Docker base image
FROM nvcr.io/nvidia/cuda:11.1-runtime-ubuntu20.04

LABEL base="nvcr.io/nvidia/cuda:11.1-runtime-ubuntu20.04"
LABEL monai="0.0-prototype.0"
LABEL name="map/spleen-segmentation"
LABEL version="0.0"

# Set dpkg to non-interactive to avoid docker build stopping to ask questions.
ENV DEBIAN_FRONTEND=noninteractive

# Set additional environment values that make usage more pleasant.
ENV TERM=xterm-256color

RUN apt update \
 && apt upgrade -y --no-install-recommends \
 && apt install -y --no-install-recommends \
    build-essential=12.8ubuntu1.1 \
    python3=3.8.2-0ubuntu2 \
    python3-pip=20.0.2-5ubuntu1.5 \
    python3-setuptools=45.2.0-1 \
    curl=7.68.0-1ubuntu2.5 \
 && apt autoremove -y \
 && rm -f /usr/bin/python /usr/bin/pip \
 && ln -s /usr/bin/python3 /usr/bin/python \
 && ln -s /usr/bin/pip3 /usr/bin/pip

RUN mkdir -p /etc/monai/
RUN mkdir -p /opt/monai/app/
RUN mkdir -p /opt/monai/executor/
RUN mkdir -p /var/opt/monai/models/

# Copy app requirements file.
COPY ./app/requirements.txt /opt/monai/app/

RUN pip install -r /opt/monai/app/requirements.txt

# Copy app files.
COPY ./app/ /opt/monai/app/

# Copy MAP files.
COPY ./app.json /etc/monai/
COPY ./pkg.json /etc/monai/
COPY ./monai-exec /opt/monai/executor/

# Copy app model(s).
COPY ./spleen_model.ts /var/opt/monai/models/spleen_model/data.ts

ENTRYPOINT [ "/opt/monai/executor/monai-exec" ]