import gzip
from textstat.textstat import textstat
from nltk import word_tokenize
from tagger import raubt_tagger
# from sklearn.svm import SVR
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split as split
from sklearn.metrics import mean_squared_error as mse
# from collections import Counter
# import pickle


def parse(filename):
    f = gzip.open(filename, 'rt')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos+2:]
        entry[eName] = rest
    yield entry

datafile = "data/books.txt.gz"


def take(n, iterator):
    for i, d in enumerate(iterator):
        if i == n:
            break
        yield d


def comparative_ratio(text):
    tokens = word_tokenize(text)
    tags = raubt_tagger.tag(tokens)
    return sum(p[1] == 'JJR' or p[1] == 'RBR' for p in tags) / len(tokens) if tokens else 0


def featurize(r):
    length = len(r['review/text']) / 1000
    lenght_vec = vectorize_length(r)
    rating_vec = vectorize_rating(r)
    readability = textstat.automated_readability_index(r['review/text'])
    comparative_index = comparative_ratio(r['review/text'])
    n, d = parse_helpfulness(r)
    y = n / d
    return [length] + lenght_vec + [readability] + rating_vec + [comparative_index, y]


def vectorize_length(r):
    n = len(r['review/text'])
    if n < 100:
        return [1, 0, 0]
    elif n < 1000:
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def vectorize_rating(r):
    rating = int(float(r['review/score'])) - 1
    rating_vec = 5 * [0]
    rating_vec[rating] = 1
    return rating_vec


def flatten(iterators):
    for iterator in iterators:
        for r in iterator:
            yield r


def product_reviews(reviews):
    r0 = next(reviews)
    product_id = r0['product/productId']
    product_reviews = [r0]
    for r in reviews:
        try:
            if r['product/productId'] == product_id:
                product_reviews.append(r)
            else:
                yield product_reviews
                product_id = r['product/productId']
                product_reviews = [r]
        except KeyError:
            pass

    yield product_reviews


def merge_duplicates(rlist):
    '''rlist is a list of reviews for a single product.'''
    key = ['review/userId', 'review/time', 'product/productId']
    keyify = lambda r: tuple(r[k] for k in key)
    reviews = {}
    for r in rlist:
        k = keyify(r)
        if k in reviews:
            new_helpfulness = add_helpfulness(r, reviews[k])
            reviews[k]['review/helpfulness'] = new_helpfulness
        else:
            reviews[k] = r

    for r in reviews.values():
        yield r


def parse_helpfulness(r):
    s = r['review/helpfulness']
    n, d = s.split('/')
    return int(n), int(d)


def has_n_helpfulness_ratings(r, N):
    n, d = parse_helpfulness(r)
    return d >= N


def add_helpfulness(r1, r2):
    n1, d1 = parse_helpfulness(r1)
    n2, d2 = parse_helpfulness(r2)
    return '{0}/{1}'.format((n1 + n2), (d1 + d2))


def has_helpfulness(r):
    return 'review/helpfulness' in r and r['review/helpfulness'] != '0/0'


def reviews(num=None, users=None, f=None, pickled=False):
    '''Returns an iterator of all Amazon reviews that have helpfullness.
    num: Limits the number of reviews. If None then all reviews.
    users: Limits to a list of users. All if None.
    f: function that is applied on each review.'''
    if pickled:
        reviews = unpickle_reviews('reviews5.pickle')
    else:
        data = parse('data/books.txt.gz')
        data = (d for d in data if has_helpfulness(d))
        rlists = product_reviews(data)
        merged = (merge_duplicates(rlist) for rlist in rlists)
        reviews0 = flatten(merged)
        reviews = (r for r in reviews0 if has_n_helpfulness_ratings(r, 5))
    if users:
        users = set(users) if type(users) is not set else users
        reviews = (r for r in reviews if r['review/userId'] in users)
    if f is not None:
        reviews = (f(r) for r in reviews)
    if num:
        reviews = take(num, reviews)
    return reviews


def unpickle_reviews(fname):
    f = open(fname, 'rb')
    try:
        while True:
            yield pickle.load(f)
    except:
        f.close()
        raise StopIteration


def extract_features(r):
    return float(r['review/score']), len(r['review/text']), 1


def data(n_samples, test_ratio, feature_extractor=extract_features):
    X = np.array([feature_extractor(r) for r in reviews(num=n_samples)])
    X_train, X_test = split(X, test_size=test_ratio)

    y = np.array([extract_labels(r) for r in reviews(num=n_samples)])
    y_train, y_test = split(y, test_size=test_ratio)

    return X_train, X_test, y_train, y_test


def error(clf, X_test, y_test):
    y_predict = clf.predict(X_test)
    return mse(y_test, y_predict)


def extract_labels(r):
    n, d = parse_helpfulness(r)
    return n / d


def pickle_data(books_zip_path):
    '''This might take a long time.'''
    data = parse(books_zip_path)
    data = (d for d in data if has_helpfulness(d))
    rlists = product_reviews(data)
    # Doesn't remove all duplicates.
    merged = (merge_duplicates(rlist) for rlist in rlists)
    reviews0 = flatten(merged)
    reviews = (r for r in reviews0 if has_n_helpfulness_ratings(r, 5))

    count5 = 0
    f = open('reviews5.pickle', 'wb')
    for r in reviews:
        count5 += 1
        pickle.dump(r, f, protocol=4)
    print('Pickled {} reviews.'.format(count5))
    f.close()


## Pickle script utilities.
# XY = np.array([featurize(r) for r in take(100000, unpickle_reviews('reviews5.pickle'))], dtype=float)

# f = open('data100K.pickle', 'wb')
# pickle.dump(XY, f, protocol=4)
# f.close()

# f = open('data100K.pickle', 'rb')
# XY = pickle.load(f)
# f.close()
# XY.shape
