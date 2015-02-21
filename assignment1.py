import gzip
# from sklearn.svm import SVR
import numpy as np
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


# def featurize(d):
#     features = ['review/time', 'review/helpfulness']
#     if all(f in d for f in features):
#         return (int(d['review/time']), d['review/helpfulness'])
#     else:
#         return None


# def find_duplicate():
#     features = ['review/userId', 'review/time', '0517150328']
#     data = (r for r in parse("data/books.txt.gz") if all(f in r for f in features))
#     ids = {}
#     extract_feature = lambda r: tuple(r[f] for f in features)
#     for r in data:
#         id0 = extract_feature(r)
#         if id0 in ids:
#             return r, ids[id0]
#         else:
#             ids[id0] = r


def flatten(iterators):
    for iterator in iterators:
        for r in iterator:
            yield r


def product_reviews(reviews):
    r0 = next(reviews)
    product_id = r0['product/productId']
    product_reviews = [r0]
    for r in reviews:
        if r['product/productId'] == product_id:
            product_reviews.append(r)
        else:
            yield product_reviews
            product_id = r['product/productId']
            product_reviews = [r]

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


def reviews(num=None, users=None, f=None):
    '''Returns an iterator of all Amazon reviews that have helpfullness.
    num: Limits the number of reviews. If None then all reviews.
    users: Limits to a list of users. All if None.
    f: function that is applied on each review.'''
    data = parse('data/books.txt.gz')
    data = (d for d in data if has_helpfulness(d))
    rlists = product_reviews(data)
    merged = (merge_duplicates(rlist) for rlist in rlists)
    reviews0 = flatten(merged)
    reviews = (r for r in reviews0 if has_n_helpfulness_ratings(r, 10))
    if users:
        users = set(users) if type(users) is not set else users
        reviews = (r for r in reviews if r['review/userId'] in users)
    if f is not None:
        reviews = (f(r) for r in reviews)
    if num:
        reviews = take(num, reviews)
    return reviews


def extract_features(r):
    return float(r['review/score']), len(r['review/text']), 1


def data(n_samples, test_ratio, feature_extractor=extract_features):
    X = np.array([feature_extractor(r) for r in reviews(num=n_samples)])
    X_train, X_test = split(X, test_size=test_ratio)

    y = np.array([extract_labels(r) for r in reviews(num=n_samples)])
    y_train, y_test = split(y, test_size=split_ratio)

    return X_train, X_test, y_train, y_test


def error(clf, X_test, y_test):
    y_predict = clf.predict(X_test)
    return mse(y_test, y_predict)


def extract_labels(r):
    n, d = parse_helpfulness(r)
    return n / d
