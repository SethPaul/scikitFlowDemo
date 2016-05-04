
# coding: utf-8

# # Titanic Kaggle Competition
#
# This dataset contains data on the passengers of the Titanic, including whether or not they survived. The goal is to be able to predict whether a passenger survived from the data we have on them.
#
# * Is this a supervised or unsupervised problem?
# * Is this a classification or regression problem?
#
# We're going to be focusing on the data exploration and preparation steps in this notebook. While this is a relatively simple dataset, it still has issues that need to be cleaned up and prepared to be used in a machine learning algorithm. We'll be doing the following:
#
# * Explore the dataset to identify potential problems and patterns
# * Deal with missing data points
# * Identify categorical variables and convert them
# * Manipulate certain features to get more information out of them

# In[4]:

import pandas as pd
import numpy as np
# Use pandas to read the csv into a DataFrame object
passengers = pd.read_csv('../titanic/data/train.csv')
test = pd.read_csv('../titanic/data/test.csv')
# Get stats on the dataset
print "\n--------First n rows---------\n"
print passengers.head()
print "\n--------Stats---------\n"
print passengers.describe()
print passengers.shape


# ## Missing Values
#
# Do you see how Age only has a count of 714 in the "describe()" output? See those "NaN" values in the "Age" and "Cabin" columns? That means we have missing data points. We need to something about that.
#
# What is a good value to fill missing "Age" values? A good choice would be the median value of our "Age" column, since that falls right in the middle of all ages in our dataset.
#
# What about the "Cabin" column? Since this isn't a numerical value we can't take its median. We'll come back to this column later in the notebook.
#
# The "Embarked" column has NaN values as well, but due to a cool trick with handling categorical variables we won't need to fill it.

# In[6]:

import matplotlib.pyplot as plt

passengers["Age"].plot(kind='hist')


# In[7]:

# pandas "fillna" lets us fill specified columns of the dataframe with a value of our choice
# http://pandas.pydata.org/pandas-docs/version/0.17.1/generated/pandas.DataFrame.fillna.html
passengers["Age"] = passengers["Age"].fillna(passengers["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# Notice how the "NaN" value from above is now 28, which is the median Age value
print passengers.head()


# ## Categorical Data
#
# Continuous numerical data is easy to work with, like "Fare" or "Age". However, what do we do with text data, or numerical data that represents a class or category, like Pclass? Let's assume "Name" and "Ticket" won't give us any meaningful insight as to whether the passenger survived. However, "Sex" and "Pclass" may be very important to whether the passenger survived. We need to convert those columns into values that machine learning algorithms can handle correctly.
#
# We will represent multiple category categorical variables like Pclass into a "one-hot" vector representation where it is a vector of binary values (1 or 0) in which only one bit is set to 1. So for example, since we have 3 possible values for Pclass (1, 2, or 3), a value of 1 would be represented as (1, 0, 0). A value of 2 would be represented as (0, 1, 0), and so on.
#
# Remember how the "Embarked" column has NaN values? Due to the fact that we're encoding its values as a "one-hot" vector we don't actually need to fill them. The one-hot representation of "Embarked" has 3 possible values (C, S, Q) where the value is the column that has a 1. However, when the value of "Embarked" is NaN, none of those values will be 1, so the vector will be (0, 0, 0), which is uniquely identifiable. Cool trick, huh?
#
# It's worth mentioning here that we could also handle categorical values by simpling assigning an integer label to each possible value - this is known as Label Encoding, as opposed to the One-Hot Encoding we're using. It's worth trying out both encoding types on your data and seeing which one performs better. I included code for label encoding as well but kept it commented out so it doesn't get confusing.

# In[8]:

# def label_encode_column(df, col):
#     col_vals = df[col].unique()
#     print col_vals
#     col_map = {col_vals[i]:i for i in range(len(col_vals))}
#     return df.apply(lambda row: col_map[row[col]], axis=1)

# Drop "Name", "Ticket" and "PassengerId" columns
passengers = passengers.drop(["Name","Ticket","PassengerId"], axis=1)
test = test.drop(["Name","Ticket","PassengerId"], axis=1)
print "----------No Name or Ticket-----------\n"
print passengers.head()

# # Make copy for label encoding
# passengers_label = passengers.copy()
# passengers_label["Embarked"] = label_encode_column(passengers_label, "Embarked")
# passengers_label["Pclass"] = label_encode_column(passengers_label, "Pclass")

# pandas "get_dummies" function let's us convert categorical columns to "one-hot" vector data
# that machine learning algorithms like
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
passengers = pd.get_dummies(passengers, columns=["Embarked","Pclass"])
test = pd.get_dummies(test, columns=["Embarked","Pclass"])

# Since "Sex" is a binary variable (only 2 possible values),
# we can simply assign numerical labels of 0 and 1 to "male" and "female"
passengers["Sex"][passengers["Sex"] == 'male'] = 0
passengers["Sex"][passengers["Sex"] == 'female'] = 1
test["Sex"][test["Sex"] == 'male'] = 0
test["Sex"][test["Sex"] == 'female'] = 1

print "\n----------Categorical Data-----------\n"
print passengers.head()


# # Label Encoding Version
# passengers_label["Sex"] = passengers["Sex"]


# ## That Pesky Cabin Column
#
# You may be wondering, why didn't we just throw out the "Cabin" column like we did with "Name" and "Ticket"? Let's think about what the Cabin value means about our passenger. Each Cabin value has a letter, representing the deck, and a room number. How could these values help us? Check out this ship plan of the Titanic:
#
# http://home.online.nl/john.vanderree/plan.htm#Decke
#
# You can see that higher deck letters are further belowdecks. Consider that the Titanic struck the iceberg at night, meaning that many passengers were likely in their rooms at the time of the crash. That being the case, the deck letter could be significant because the higher the deck letter the further belowdecks the passengers were at the time of the crash (assuming they were in their rooms). This could have affected how quickly they got to the lifeboats and therefore their survival chances. While the room number could have some effect on how quickly they could get to lifeboats, it is probably not as significant as the deck letter, so we'll ignore it.
#
# One big caveat here is that the deck letter is probably highly correlated with the passenger class. We won't go into it in-depth here, but know that if one of your features is correlated with another, it's not going to add much extra information and may even hurt your predictions. We're going to ignore the possible correlation here for the sake of learning how to manipulate our columns into new features, but keep that in mind. Another caveat is that most of the cabin values are missing, meaning that we may not get much signal out of it.
#
# Let's parse out the deck letter from each "Cabin" value and encode the column to a one-hot vector. As with "Embarked" above, we don't need to fill the NaN values due to the cool aspect of one-hot encoding.

# In[9]:

# function for extracting deck letter
def get_deck_letter(row):
    # Ignore NaN values
    if not pd.isnull(row["Cabin"]):
        # Get first letter of "Cabin" value
        return str(row["Cabin"])[0]
    # Otherwise return NaN
    return row["Cabin"]

# # Label Version
# passengers_label["Cabin"] = passengers_label.apply(lambda row: get_deck_letter(row), axis=1)
# passengers_label["Cabin"] = label_encode_column(passengers_label, "Cabin")

# set cabin values to deck letter
passengers["Cabin"] = passengers.apply(lambda row: get_deck_letter(row), axis=1)
test["Cabin"] = test.apply(lambda row: get_deck_letter(row), axis=1)
print "--------Filled Cabin-----------\n"
print passengers.head()
print "\n\n"

passengers = pd.get_dummies(passengers, columns=["Cabin"])
test = pd.get_dummies(test, columns=["Cabin"])
print "--------Categorical Cabin-----------\n"
print passengers.head()


# ## More Intuition
#
# It seems likely that the first to be saved on the Titanic would be women and children. We already have a binary variable for whether the passenger was a woman, so let's create a variable that says whether the passenger is a child.
#
# Let's then see how accurate our predictions would be if we based it on just one column, like "Sex" or "IsChild".

# In[ ]:

# Function for determining if passenger is a child
def is_child(row):
    child = 0 if row["Age"] >= 12 else 1
    return child

passengers["IsChild"] = passengers.apply(lambda row: is_child(row), axis=1)
passengers = passengers.drop("Age", axis=1)
test["IsChild"] = test.apply(lambda row: is_child(row), axis=1)
test = test.drop("Age", axis=1)

test = test.fillna(0)
test['Cabin_T'] = np.zeros(len(test))
# # Label encoding version
# passengers_label["IsChild"] = passengers["IsChild"]


# In[11]:

from sklearn.metrics import confusion_matrix
import numpy as np

def predict_by_column(df, col, col_val, target_col):
    """
    df: dataframe to make predictions on
    col: predict target based on this column being equal to `col_val`
    col_val: value of col
    target_col: column you are trying to predict
    """
    y = df[target_col].values
    x = df[col].values
    predictions = []
    num_correct = 0
    for i in range(len(y)):
        pred = 1 if x[i] == col_val else 0
        predictions.append(pred)
        actual = y[i]
        if pred == actual:
            num_correct += 1
    acc = num_correct / float(len(y))
    cm = confusion_matrix(np.array(predictions).astype(int), y)
    return acc, cm

def get_best_columns(df, n, target_col):
    accuracies = []
    for c in df.columns:
        if c != target_col and len(df[c].unique()) == 2 and 1 in df[c].unique():
            acc, cm = predict_by_column(df, c, 1, target_col)
            accuracies.append((c, {"Acc":acc, "CM": cm}))
    return sorted(accuracies, key=lambda x: x[1]['Acc'], reverse=True)[:n]

best_cols = get_best_columns(passengers, 5, "Survived")
for c in best_cols:
    print c[0] + ":"
    print "\tAcc: " + str(c[1]["Acc"])
    print "\tConfusion Matrix:"
    print "\t"+str(c[1]["CM"][0])
    print "\t"+str(c[1]["CM"][1])
    print ""


# ## Machine Learning
#
# Now that we've cleaned up our data, let's get it ready to put into a machine learning model. We need to do a few things:
#
# * Extract the "Survived" column to be our prediction labels
# * Split our dataset into a training set and a testing set
# * Train a machine learning model on the training set
# * Test the trained model on our test set and evaluate its performance
#
# We split our dataset into a training set and a test set so we can have a way to evaluate how well our model will do on data it hasn't trained on. We know the answers to the test set, but the model doesn't, so we can assess its accuracy.
#
# Let's start with the first two.
#
# ### Split the Dataset

# In[12]:

from sklearn import cross_validation
# Get feature matrix and label vector
X = passengers.drop("Survived", axis=1).values
y = passengers["Survived"].values

X_sub = test.values

# Split into training and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=42)


# ### Train the Model
#
# We're going to use a Random Forest as our machine learning model. Don't worry about the details of the algorithm, we'll cover that in a later meetup.

# In[13]:

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

target_names = ["Didn't Survive", "Survived"]

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print rf.score(X_test, y_test)
# scores = cross_validation.cross_val_score(rf, X, y, cv=5, scoring='f1_weighted')
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print scores
cm = confusion_matrix(y_test, y_pred)
print cm
plot_confusion_matrix(cm)


# In[15]:

from sklearn.utils.extmath import density

def show_most_informative_features(feature_names, clf, n=20):
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

classifiers = [
    LinearSVC(),
    MultinomialNB(),
    SGDClassifier(),
    AdaBoostClassifier(),
    Perceptron(),
    PassiveAggressiveClassifier(),
    KNeighborsClassifier(),
    RandomForestClassifier()
]

feature_names = passengers.drop("Survived", axis=1).columns

pred_map = {}

for clf in classifiers:
    clf_name = clf.__class__.__name__
    print clf_name
    print ""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    pred_map[clf_name] = clf.predict(X_sub)
    print clf.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    print cm
    #plot_confusion_matrix(cm)


    died_words = []
    survived_words = []

    if hasattr(clf, 'coef_'):
        show_most_informative_features(feature_names, clf, 5)

    print "\n\n"

def save_submission(df, preds, fname):
    submission = pd.DataFrame(np.array([df.PassengerId, preds]).transpose(), columns=["PassengerId","Survived"])
    submission.to_csv(fname, index=False)
    return submission

sub = save_submission(pd.read_csv('data/test.csv'), pred_map['Perceptron'], 'perceptrons_suck.csv')
