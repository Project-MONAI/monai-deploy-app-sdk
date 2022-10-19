FROM nvcr.io/nvidia/pytorch:21.07-py3 AS foundation

ARG EXTRA_PYTHON_PACKAGES

RUN apt-get -y update && \
    apt-get -y install python3-distutils python3-pip python3-venv && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --ignore-installed setuptools

# Create a Virtual Environment to limit the size of the application container
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m pip install --upgrade pip

# Copy the ai_service wheel, this is separate from requirements.txt to help with layer caching for repeated builds
COPY lib/ai_service-*-py3-none-any.whl /tmp/
RUN python3 -m pip install --no-cache-dir /tmp/ai_service-*-py3-none-any.whl

COPY requirements.txt /tmp/
# Add any other python packages your AI Service requires
RUN python3 -m pip install --no-cache-dir ${EXTRA_PYTHON_PACKAGES} -r /tmp/requirements.txt

FROM nvcr.io/nvidia/pytorch:21.07-py3 AS application

ARG PARTNER_NAME
ARG SERVICE_NAME
ARG VERSION
ARG MONAI_APP_MODULE
ARG MODEL_PATH

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# python3-gdcm or python-gdcm is required for decompression
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends python3-gdcm && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# copy model file to model folder
RUN wget -q https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/lung_nodule_ct_detection_v0.2.0.zip && \
    unzip lung_nodule_ct_detection_v0.2.0.zip -d /tmp/ && \
    mkdir -p /app/model && \
    cp /tmp/lung_nodule_ct_detection/models/model.ts /app/model/ && \
    rm -rf /tmp/lung_nodule_ct_detection && \
    rm lung_nodule_ct_detection_v0.2.0.zip

# non-root aiserviceuser in group aiserviceuser with UserID and GroupID as 20225
RUN groupadd -g 20225 -r aiserviceuser && \
    useradd -u 20225 -r -g aiserviceuser aiserviceuser && \
    chown -R aiserviceuser:aiserviceuser /app /var
USER aiserviceuser:aiserviceuser

# Enable Matplotlib cache folder
RUN mkdir -p /app/.config/matplotlib
ENV MPLCONFIGDIR=/app/.config/matplotlib

# Copy the virtual environment from the foundation image
ENV VIRTUAL_ENV=/app/venv
COPY --from=foundation --chown=aiserviceuser:aiserviceuser /opt/venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# make sure all messages reach the console
ENV PYTHONUNBUFFERED=1

# copy MONAI app files
COPY --chown=aiserviceuser:aiserviceuser app_wrapper.py /app/
COPY --chown=aiserviceuser:aiserviceuser app/* /app/app/
WORKDIR /app

ENV AI_PARTNER_NAME ${PARTNER_NAME}
ENV AI_SVC_NAME ${SERVICE_NAME}
ENV AI_SVC_VERSION ${VERSION}
ENV AI_MODEL_PATH ${MODEL_PATH}
ENV MONAI_APP_CLASSPATH ${MONAI_APP_MODULE}

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ENV DEBUG=NO
ENV KEEP_FILES=NO

CMD ["python", "app_wrapper.py"]
