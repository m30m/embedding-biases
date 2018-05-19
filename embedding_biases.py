import json
import random

from flask import Flask, request, render_template, jsonify
import numpy as np

app = Flask(__name__)


def read_model(path):
    model = {}
    for line in open(path):
        if not line:
            continue
        splitted = line.split()
        token = splitted[0]
        dims = map(float, splitted[1:])
        dims = np.array(list(dims), dtype=np.float32)
        model[token] = dims
    return model


embedding_model = read_model('small.txt')


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def get_embedding(word):
    return embedding_model.get(word)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/templates')
def templates():
    return jsonify(json.loads(open('word_templates.json').read()))


@app.route('/exists')
def word_exists():
    word = request.args.get('word')
    if get_embedding(word) is not None:
        return 'Yes!'
    return 'No!'


def association(x, A, B):
    return np.mean([cos_sim(x, a) for a in A]) - np.mean([cos_sim(x, b) for b in B])


def p_value(X, Y, A, B):
    X = [association(x, A, B) for x in X]
    Y = [association(y, A, B) for y in Y]
    default_sum = sum(X) - sum(Y)
    union = X + Y
    total_sum = sum(union)
    half = int(len(union) / 2)
    EXPS = 10000
    true_exps = 0
    random.seed(32)
    for i in range(EXPS):
        random.shuffle(union)
        new_X = union[:half]
        if 2 * sum(new_X) - total_sum > default_sum:
            true_exps += 1
    return true_exps / EXPS


@app.route('/weat', methods=['POST'])
def association_test():
    request_json = request.get_json(force=True)
    X = list(map(get_embedding, request_json['target_X']))
    Y = list(map(get_embedding, request_json['target_Y']))
    A = list(map(get_embedding, request_json['attribute_A']))
    B = list(map(get_embedding, request_json['attribute_B']))
    return str(p_value(X, Y, A, B))


if __name__ == '__main__':
    app.run()
