FROM python:3.9-slim

RUN pip install pipenv

WORKDIR app
COPY ["Pipfile", "Pipfile.lock", "/app/"]

RUN pipenv install --system --deploy

COPY ["scripts", "/app/scripts/"]
COPY ["artifacts", "/app/artifacts/"]
COPY ["data", "/app/data/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:9696", "scripts.model_evaluation:app"]