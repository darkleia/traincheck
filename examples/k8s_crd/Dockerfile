FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["torchrun"]
