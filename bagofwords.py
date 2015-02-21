import nltk
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
import re
#import repo.assignment1 as dm
import assignment1 as dm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def review_to_words( review_text ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  


def get_bag_of_words(x, vocabulary = None):
  num_reviews = len(x)
  clean_reviews = []
  # Loop over each review; create an index i that goes from 0 to the length
  # of the movie review list 
  for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_reviews.append( review_to_words( x[i] ) )

  print "Creating the bag of words...\n"

  # Initialize the "CountVectorizer" object, which is scikit-learn's
  # bag of words tool.  
  vectorizer = CountVectorizer(analyzer = "word",   \
                               tokenizer = None,    \
                               preprocessor = None, \
                               stop_words = None,   \
                               max_features = 5000,
                               vocabulary = vocabulary) 

  # fit_transform() does two functions: First, it fits the model
  # and learns the vocabulary; second, it transforms our training data
  # into feature vectors. The input to fit_transform should be a list of 
  # strings.
  train_data_features = vectorizer.fit_transform(clean_reviews)

  # Numpy arrays are easy to work with, so convert the result to an 
  # array
  train_data_features = train_data_features.toarray()

  print train_data_features.shape

  vocab = vectorizer.get_feature_names()
  # print vocab

  return train_data_features, vocab

# Sum up the counts of each vocabulary word
#dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
#for tag, count in zip(vocab, dist):
#  print count, tag


def extract_text_features(r):
  return r['review/text']


def data(n_samples, split_ratio):
  return dm.data(n_samples, split_ratio, extract_text_features)
