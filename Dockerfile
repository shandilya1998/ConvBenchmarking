FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
WORKDIR /root

ADD conv ./conv
COPY requirements.txt ./requirements.txt
COPY main.py ./main.py
COPY run.py ./run.py
COPY test.py ./test.py

RUN pip install -r requirements.txt
RUN wandb login 4a30a34490a130dc21329fd04548bfd9f01cb1ec
ENTRYPOINT ["python", "-u", "main.py"]
