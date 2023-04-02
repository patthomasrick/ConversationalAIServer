FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy main.
COPY src/ChatServer /app/ChatServer

RUN mkdir -p /app/data

CMD ["flask", "--app=ChatServer.server", "run", "--host=0.0.0.0"]
