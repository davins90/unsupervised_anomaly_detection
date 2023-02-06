FROM jupyter/datascience-notebook:python-3.9.2

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "_🏡_Homepage.py" , "--server.port=8501", "--server.address=0.0.0.0"]
# ENTRYPOINT ["streamlit", "run", "_🏡_Homepage.py" ]