FROM python:3.8-slim-buster 

EXPOSE 5000

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY ./requirements.txt /var/www/App/requirements.txt
RUN python -m pip install -r /var/www/App/requirements.txt 

RUN pip install --ignore-installed six watson-developer-cloud
RUN pip install soundfile
WORKDIR /app
COPY . /app
ENV NUMBA_CACHE_DIR=/tmp/numba_cache


RUN useradd appuser && chown -R appuser /app
USER appuser

CMD ["python","app.py"]
