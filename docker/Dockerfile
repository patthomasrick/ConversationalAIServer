FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Setup just grabs the models.
COPY src/setup.py /app/
ENV MODEL="facebook/blenderbot-400M-distill"
RUN python setup.py

# Copy main.
COPY src/*.py /app/

RUN mkdir -p /app/data

CMD ["flask", "--app=ChatsServer.server", "run", "--host=0.0.0.0"]
