FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4
WORKDIR /root

ADD conv ./conv
COPY requirements.txt ./requirements.txt
COPY main.py ./main.py
COPY run.py ./run.py
COPY test.py ./test.py

RUN pip install -r requirements.txt
ENTRYPOINT ["python", "-u", "main.py"]
