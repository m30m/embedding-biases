# Embedding Biases

This project is just a visualization for [this](http://opus.bath.ac.uk/55288/4/CaliskanEtAl_authors_full.pdf) paper.

# Setup

## In-memory
You can use the in-memory option for storing the vectors embeddings. This is good if you just want to try out this app:

```sh
python3 embedding_biases.py --vectors small.txt
```

## Redis
This option use an already populated redis database for getting word embeddings.

```sh
python3 embedding_biases.py --storage redis
```