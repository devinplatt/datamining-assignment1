import gzip
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


def featurize(d):
    features = ['review/time', 'review/helpfulness']
    if all(f in d for f in features):
        return (int(d['review/time']), d['review/helpfulness'])
    else:
        return None


def find_duplicate():
    features = ['review/userId', 'review/time', '0517150328']
    data = (r for r in parse("data/books.txt.gz") if all(f in r for f in features))
    ids = {}
    extract_feature = lambda r: tuple(r[f] for f in features)
    for r in data:
        id0 = extract_feature(r)
        if id0 in ids:
            return r, ids[id0]
        else:
            ids[id0] = r


def flatten(iterators):
    for iterator in iterators:
        for r in iterator:
            yield r


def product_reviews(reviews):
    r = next(reviews)
    product_id = r['product/productId']
    product_reviews = [r]
    for r in reviews:
        if r['product/productId'] == product_id:
            product_reviews.append(r)
        else:
            yield product_reviews
            product_id = r['product/productId']
            product_reviews = [r]

    yield product_reviews


def merge_duplicates(rlist):
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
    yield from reviews.values()


def parse_helpfulness(r):
    s = r['review/helpfulness']
    n, d = s.split('/')
    return int(n), int(d)


def add_helpfulness(r1, r2):
    n1, d1 = parse_helpfulness(r1)
    n2, d2 = parse_helpfulness(r2)
    return '{0}/{1}'.format((n1 + n2), (d1 + d2))
