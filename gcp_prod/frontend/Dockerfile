FROM jupyter/datascience-notebook:python-3.9.2

EXPOSE 8080

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

RUN streamlit run _🏡_Homepage.py 

WORKDIR .
COPY . .
