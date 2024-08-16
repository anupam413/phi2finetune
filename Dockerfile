FROM python:3.8.10

RUN mkdir my-model
ENV MODEL_DIR=/home/phi2_finetune/my-model
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib

COPY requirements.txt ./requirements.txt
RUN pip install joblib
RUN pip install -r requirements.txt

COPY train_data.jsonl ./train_data.jsonl
COPY test_data.jsonl ./test_data.jsonl

COPY phi2_train.py ./train.py
COPY phi2_inference.py ./inference.py

RUN python3 train.py

