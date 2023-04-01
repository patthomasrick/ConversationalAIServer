FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Setup just grabs the models.
COPY src/setup.py /app/
RUN python setup.py

# Copy main.
COPY src/*.py /app/

RUN mkdir -p /app/data

CMD ["flask", "--app=main", "run", "--host=0.0.0.0"]
