FROM jupyter/datascience-notebook:python-3.9.2

COPY requirements_fastapi.txt .
RUN pip install -r requirements_fastapi.txt

# WORKDIR /src
# COPY . /src

COPY . . 

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

