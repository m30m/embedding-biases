# Embedding Biases

This project is just a visualization for [this](http://opus.bath.ac.uk/55288/4/CaliskanEtAl_authors_full.pdf) paper.

# Installation

Install the dependencies:

```sh
pip3 install -r requirements.txt
```

You can run the server in two modes:

## In-memory Usage
You can use the in-memory option for storing the vectors embeddings. This is good if you just want to try out this app:

```sh
python3 embedding_biases.py --vectors small.txt
```

## Redis Usage
This option use an already populated redis database for getting word embeddings.
You can easily spawn a redis container via the following commands:

```sh
apt install docker.io
docker run -d --name embedding-redis -p 127.0.0.1:6379:6379 redis
```


Then you should setup the redis with embedding vectors:

```sh
python3 embedding_biases.py --storage redis --redis-url redis://127.0.0.1:6379/ --fill-redis
```

And then you can start the server:

```sh
python3 embedding_biases.py --storage redis --redis-url redis://127.0.0.1:6379/
```