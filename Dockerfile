FROM nvcr.io/nvidia/pytorch:26.08-py3
#FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel

WORKDIR /workspace

COPY requirements.txt /workspace

RUN pip install --no-cache-dir -r requirements.txt

COPY deps.sh /workspace

RUN chmod +x /workspace/deps.sh

RUN ./deps.sh

COPY gda-qa-lora /workspace/gda-qa-lora

COPY dataset /workspace/dataset

COPY eval.py /workspace