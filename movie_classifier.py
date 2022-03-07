# -*- coding: utf-8 -*-


import csv
import random
### Answer starts here ###
import re
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
### Answer ends here ###

"""Let's download the dataset:"""

!wget https://raw.githubusercontent.com/McGillAISociety/BootcampAssignmentDatasets/master/data/assignment2/train_reviews.csv

"""And create a function to print a review:"""

def print_review(review, score):
  print('--------------- Review with score of {} ---------------'.format(score))
  print(review)
  print('------------------------------------------------------')
  print()

"""Let's load the data and see what the first 10 reviews look like:

"""

with open('train_reviews.csv') as csv_file:
  csv_reader = csv.reader(csv_file)
  colnames = next(csv_reader)  # skip column names
  data = list(csv_reader)

for review, score in random.sample(data, 10):
  print_review(review, score)

"""## 2. Preprocessing
 We will be converting our data into a binary bag-of-words representation (Google "binary bag-of-words"). To do this, we will perform two steps beforehand.

### Question Cleaning the train data
Create a function called `clean`, which takes a string and then:

 1. lower-case all words 
 2. only keeps letters and spaces
 

 We also need to get rid of [HTML tags](https://www.javatpoint.com/html-tags) as they do not hold valuable information for classifying the review. A quick Google search on removing HTML tags with `regular expressions` will show you how to do this! 
  

"""

def clean(review):
  ### Start of Answer ###
  cleantext = re.sub(re.compile('<.*?>'), '', review).lower()
  cleantext = re.sub(r'[^a-z ]+', '', cleantext)
  return cleantext
  
  ### End of Answer ###



print(clean("This was the WORST movie I have EVER SEEN!! <br/> "))


X_train = []
y_train = []
for review, score in data:
  X_train.append(clean(review))
  y_train.append(int(score))

"""###  Picking features


"""

def get_vocab(reviews, vocab_size):
  ### Answer starts here ###
  
  word_list = []

  for review in reviews:
    for word in review.split(" "):
      word_list.append(word)

  count = Counter(word_list)
  c = [a_tuple[0] for a_tuple in count.most_common(vocab_size)]

  return c
  

  ### Answer ends here ###

"""Test your function with the following code. The `vocabulary` variable should have a length of 10,000 and the most common words should be "the", "and", "a", etc."""

num_features = 10000
vocabulary = get_vocab(X_train, num_features)
print(len(vocabulary))
print(vocabulary)

def vectorize(review_string, vocab):
  ### Answer starts here ###
  vector = []
  review_words = review_string.split(" ")

  for i in vocab:
    if i in review_words:
      vector.append(1)
    else:
      vector.append(0)
    
  return vector
  
  ### Answer ends here ###

"""Test your function with the following input. The vector should have four "1"s."""

vector = vectorize("the and a of zyxw", vocabulary)
print(vector)
print(sum(vector))

"""Now, vectorize the whole dataset. """

### Answer starts here ###

X_train_vect = []

for i in X_train:
  X_train_vect.append(vectorize(i, vocabulary))


### Answer ends here ###

for i in range(5):
  print_review(X_train_vect[i], y_train[i])

"""For convenience, we will write a function called `preprocess_sample_point` which takes as input a single raw review and ouputs its binary bag-of-words representation."""

def preprocess_sample_point(review, vocab):
  return vectorize(clean(review), vocab)

vectorized_review = preprocess_sample_point(
    'The movie was not bad, it was really good!', vocabulary)
print(sum(vectorized_review))
print(vectorized_review)

"""###Preparing the test set

Now that we have defined a cleaning function and extracted the features from the train set, we are ready to preprocess the test set. Implement the `preprocess` function below such that it:

1. Loads the raw data from a csv file 
2. Cleans and vectorizes the reviews
3. Converts the scores to `int`
4. Returns the data into a  `(X_test, y_test)` tuple
"""

def preprocess(csv_filename, vocab):
  ### Answer starts here ###
  with open(csv_filename) as csv_file:
    csv_reader = csv.reader(csv_file)
    colnames = next(csv_reader)  # skip column names
    data = list(csv_reader)
  
  X = []
  y = []
  for review, score in data:
    X.append(vectorize(clean(review), vocab))
    y.append(int(score))

  return (X, y)
  ### Answer ends here ###

!wget https://raw.githubusercontent.com/McGillAISociety/BootcampAssignmentDatasets/master/data/assignment2/test_reviews.csv

X_test, y_test = preprocess('test_reviews.csv', vocabulary)

for i in range(5):
  print_review(X_test[i], y_test[i])

"""##3. Naive Bayes


Naive Bayes classifiers are part of a larger family of classifiers which are called 'probabilistic classifiers': not only do they try to predict classes given features, but they also estimate probability distributions over a set of classes.


**Definition:** A *prior probability* is the likelihood of an event given no further assumptions. For instance, the probability that it's raining is relatively low.

**Definiton:** A *posterior probability* or *conditional probability* is the likelihood of an event given that some other event has occurred. For instance, the probability that it's raining given that there are clouds is higher than if we don't make that assumption.


### Estimating the Probability Distribution

"""

### Answer starts here ###

#Set of vectors with review 1
p_1 = []
#Set of vectors with review -1
p_2 = []

for i,j in zip(X_train_vect, y_train):
  if j==1:
    p_1.append(i)
  else:
    p_2.append(i)

#Number of negative & positive reviews
pos_reviews_count=(len(p_1))
neg_reviews_count=(len(p_2))

#Transform them into numpy arrays
p_1 = np.asarray(p_1)
p_2 = np.asarray(p_2)

a=p_1.T[48].sum()
b=p_2.T[48].sum()

#Calculate probabilities of good/bad reviews for each word
probabilities_1 = []
probabilities_2 = []

for i in p_1.T:
  probabilities_1.append(i.sum()/a)

for i in p_2.T:
  probabilities_2.append(i.sum()/b)



"""###

Create a function called `naive_bayes` which will take as input a list of features $x_1, \ldots, x_n$ and outputs the class with the largest posterior probability given the input features.
"""

def naive_bayes(vec):
  ### Answer starts here ###
  probability_positive = 1
  probability_negative = 1

  for i,j1,j2 in zip(vec, probabilities_1, probabilities_2):
    if i==1:
      probability_positive = probability_positive * j1
      probability_negative = probability_negative * j2
  if probability_positive >= probability_negative:
    return 1
  else:
    return -1


"""###  Measuring Performance

Using the naive Bayes classifier, predict the classes for each sample point in the training set as well as the test set and print accuracies.

"""


pred = []

for i in X_test:
  pred.append(naive_bayes(i))

print("accuracy is " + str(accuracy_score(y_test, pred)))


print(naive_bayes(preprocess_sample_point(
    'Terrible. Horrible. Boring. This movie is bad', vocabulary)))

print(naive_bayes(preprocess_sample_point(
    'This movie was pretty good', vocabulary)))

"""## 4. Support Vector Machines

Quick recap of SVM: A support vector classifier tries to find the best separating hyperplane through the data. If the data is linearly separable, it finds a hyperplane that maximizes the margin. If it isn't, the classifier tries to minimize the cost associated with misclassifying points.

"""


svm_clf = LinearSVC()

svm_clf.fit(X_train_vect, y_train)

y_pred = svm_clf.predict(X_test)

accuracy_score(y_test, y_pred)


print(svm_clf.predict([preprocess_sample_point(
    'Boring. Such a bad movie. It was terrible and predictable', vocabulary)]))

print(svm_clf.predict([preprocess_sample_point(
    'I really liked this movie, it\'s great!', vocabulary)]))

"""## 5. Random Forests

Random forests are a type of ensemble classifier, i.e. they are made up of a number of 'weak' learners where the final classification is a combination of the classifications of each learner.
"""



rfc = RandomForestClassifier(n_estimators=300, min_samples_split=4)

rfc.fit(X_train_vect, y_train)

y_pred = rfc.predict(X_test)

accuracy_score(y_test, y_pred)

print(rfc.predict([preprocess_sample_point(
    'Boring. This movie is terrible', vocabulary)]))

print(rfc.predict([preprocess_sample_point(
    'This movie was pretty good', vocabulary)]))

