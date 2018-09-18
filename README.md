# Embedding Biases

This project is just a visualization for [this](http://opus.bath.ac.uk/55288/4/CaliskanEtAl_authors_full.pdf) paper.

You can test arbitrary word groups to see if they are related to each other in any biased way or not.

# Installation

Install the dependencies:

```sh
pip3 install -r requirements.txt
```

You can run the server in three modes:

## In-memory Usage
You can use the in-memory option for storing the vectors embeddings. This is good if you just want to try out this app:

```sh
python3 embedding_biases.py --vectors small.txt
```

## Sqlite Usage
This option use an already populated sqlite database for getting word embeddings.

First you should create and fill the sqlite db:

```sh
python3 embedding_biases.py --storage sqlite --sqlite-path sqlite.db --fill-db
```

And then you can start the server:

```sh
python3 embedding_biases.py --storage sqlite --sqlite-path sqlite.db
```

## Redis Usage
You can use redis instead of sqlite for faster access.
You can easily spawn a redis container via the following commands:

```sh
apt install docker.io
docker run -d --name embedding-redis -p 127.0.0.1:6379:6379 redis
```


Use the following command for filling up redis:

```sh
python3 embedding_biases.py --storage redis --redis-url redis://127.0.0.1:6379/ --fill-db
```

And then you can start the server:

```sh
python3 embedding_biases.py --storage redis --redis-url redis://127.0.0.1:6379/
```