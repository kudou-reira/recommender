# FROM nvidia/cuda:8.0-cudnn6-devel

FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt ./
# RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

CMD ["python3", "-u", "main.py"]