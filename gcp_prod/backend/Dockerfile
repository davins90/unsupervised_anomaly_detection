FROM jupyter/datascience-notebook:python-3.9.2

# EXPOSE 8000

ENTRYPOINT [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload" ]

COPY requirements_fastapi.txt .
RUN pip install -r requirements_fastapi.txt

WORKDIR .
COPY . .
