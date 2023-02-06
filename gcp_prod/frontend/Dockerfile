FROM jupyter/datascience-notebook:python-3.9.2

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

CMD ["streamlit", "run", "_🏡_Homepage.py" ]