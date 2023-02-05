FROM jupyter/datascience-notebook:python-3.9.2

# EXPOSE 8000

WORKDIR .

COPY requirements_fastapi.txt .
RUN pip install -r requirements_fastapi.txt

COPY . .

ENV PORT 8000
ENV HOST 0.0.0.0

EXPOSE 8000:8000

CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload" ]
