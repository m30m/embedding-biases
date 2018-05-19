import json
import optparse
import random
import struct

import numpy as np
from flask import Flask, request, render_template, jsonify
from redis import StrictRedis

app = Flask(__name__)
embedding_model = None

def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def get_embedding(word):
    return embedding_model.get(word.lower())


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
    EXPS = 100000
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
    return 'p value is %.6f' % p_value(X, Y, A, B)


def flaskrun(app):
    """
    Takes a flask.Flask instance and runs it. Parses
    command-line flags to configure the app.
    """
    global embedding_model
    global redis_client

    default_host = "127.0.0.1"
    default_port = "5000"
    default_storage = 'in_memory'
    default_redis_url = 'redis://localhost:6379/'

    # Set up the command-line options
    parser = optparse.OptionParser()
    parser.add_option("-H", "--host",
                      help="Hostname of the Flask app " + \
                           "[default %s]" % default_host,
                      default=default_host)
    parser.add_option("-P", "--port",
                      help="Port for the Flask app " + \
                           "[default %s]" % default_port,
                      default=default_port)

    parser.add_option("-S", "--storage",
                      help="Which storage to use. in_memory or redis" + \
                           "[default %s]" % default_storage,
                      default=default_storage, choices=['in_memory', 'redis'])

    parser.add_option("-V", "--vectors",
                      help="Path to embedding file")

    parser.add_option("--fill-redis", action="store_true", dest="fill_redis",
                      help="Only fill the redis database and exit", )

    parser.add_option("--redis-url",
                      help="Redis database url " + \
                           "[default %s]" % default_redis_url, default=default_redis_url)

    options, _ = parser.parse_args()

    if options.storage == 'in_memory':
        print('Using in memory')

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

        if options.vectors is None:
            print('Please specify embedding vectors path')
            parser.print_help()
            return None
        embedding_model = read_model(options.vectors)
    else:
        print('Using Redis %s' % options.redis_url)
        redis_client = StrictRedis.from_url(options.redis_url)
        if options.fill_redis:
            if options.vectors is None:
                print('Please specify embedding vectors path')
                parser.print_help()
                return None
            print('Filling redis')
            token_count = 0
            for line in open(options.vectors):
                if not line:
                    continue
                splitted = line.split()
                token = splitted[0]
                dims = list(map(float, splitted[1:]))
                redis_client.set(token, struct.pack('f' * 300, *dims))
                token_count += 1
                if token_count % 10000 == 0:
                    print('Processed %d words' % token_count)
            print('Finished filling redis with %d words' % token_count)
            return None
        else:
            class RedisStorage(object):
                def get(self, word):
                    result = redis_client.get(word)
                    if result:
                        result = struct.unpack('f' * 300, result)
                    return result

            embedding_model = RedisStorage()

    app.run(
        host=options.host,
        port=int(options.port)
    )


if __name__ == '__main__':
    flaskrun(app)
