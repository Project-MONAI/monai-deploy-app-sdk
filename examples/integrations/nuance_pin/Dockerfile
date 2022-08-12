FROM debian:buster-slim AS foundation

RUN apt-get -y update && \
    apt-get -y install libjpeg-dev python3.7 python3-pip python3-venv zlib1g-dev

# Create a Virtual Environment to limit the size of the application container
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pip Upgrade is after virtual environment configuration.
RUN python3 -m pip install --upgrade --no-cache-dir pip && \
    python3 -m pip install --upgrade --no-cache-dir --ignore-installed setuptools

COPY requirements.txt /tmp/
# Add any other python packages your AI Service requires
ARG EXTRA_PYTHON_PACKAGES

RUN python3 -m pip install --upgrade --no-cache-dir ${EXTRA_PYTHON_PACKAGES} -r /tmp/requirements.txt

COPY lib/ai_service-*-py3-none-any.whl /tmp/
RUN python3 -m pip install --upgrade --no-cache-dir /tmp/ai_service-*-py3-none-any.whl  ${EXTRA_PYTHON_PACKAGES} && \
    rm /tmp/ai_service-*-py3-none-any.whl

FROM debian:buster-slim AS application

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# python3-gdcm or python-gdcm is required for decompression
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends python3-distutils python3-gdcm && \
    apt-get -y install python3.7 && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DEBUG=YES
ENV KEEP_FILES=YES

# make sure all messages reach the console
ENV PYTHONUNBUFFERED=1

# copy and activate virtualenv
ENV VIRTUAL_ENV=/app/venv
COPY --from=foundation /opt/venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# copy MONAI app files
COPY . /app/.
WORKDIR /app

# non-root aiserviceuser in group aiserviceuser with UserID and GroupID as 20225
RUN groupadd -g 20225 -r aiserviceuser && useradd -u 20225 -r -g aiserviceuser aiserviceuser && chown -R aiserviceuser:aiserviceuser /app
USER aiserviceuser:aiserviceuser

ARG PARTNER_NAME="nvidia"
ARG SERVICE_NAME="ai_spleen_seg_app"
ARG VERSION="0.1.0"
ARG MONAI_APP_PATH

ENV AI_PARTNER_NAME ${PARTNER_NAME}
ENV AI_SVC_NAME ${SERVICE_NAME}
ENV AI_SVC_VERSION ${VERSION}
ENV MONAI_APP_CLASSPATH ${MONAI_APP_PATH}

# we have an `app.py` file in the package which we want to use and is under `/app` in the container
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
CMD ["python3", "app_wrapper.py"]
