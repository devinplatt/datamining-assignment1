import nltk.tag
from nltk.tag import brill
import nltk.corpus


conll_sents = nltk.corpus.conll2000.tagged_sents()
conll_train = list(conll_sents[:4000])

def backoff_tagger(tagged_sents, tagger_classes, backoff=None):
    if not backoff:
        backoff = tagger_classes[0](tagged_sents)
        del tagger_classes[0]

    for cls in tagger_classes:
        tagger = cls(tagged_sents, backoff=backoff)
        backoff = tagger

    return backoff

word_patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*ould$', 'MD'),
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*ness$', 'NN'),
    (r'.*ment$', 'NN'),
    (r'.*ful$', 'JJ'),
    (r'.*ious$', 'JJ'),
    (r'.*ble$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*ive$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*est$', 'JJ'),
    (r'^a$', 'PREP'),
]

raubt_tagger = backoff_tagger(conll_train, [nltk.tag.AffixTagger, nltk.tag.UnigramTagger,
                                            nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
                              backoff=nltk.tag.RegexpTagger(word_patterns))
