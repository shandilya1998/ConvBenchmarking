FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4
WORKDIR /root

COPY conv/fft.py ./conv/fft.py
COPY conv/im2col.py ./conv/im2col.py
COPY conv/naive.py ./conv/naive.py
COPY conv/winograd.py ./conv/winograd.py
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
RUN wandb login 4a30a34490a130dc21329fd04548bfd9f01cb1ec
ENTRYPOINT ["python", "-u", "train.py"]
