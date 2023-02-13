FROM continuumio/miniconda3

WORKDIR /home/app

COPY *.py .

COPY *.txt .

RUN pip install -r requirements.txt

ENTRYPOINT streamlit run app.py
