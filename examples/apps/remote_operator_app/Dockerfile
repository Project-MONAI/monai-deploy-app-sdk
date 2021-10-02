FROM python:3.7-slim

WORKDIR /app

COPY docker_median.py app.py
RUN pip3 install scikit-image

CMD ["python3", "app.py"]
