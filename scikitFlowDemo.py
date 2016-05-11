# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sklearn as sk
import pandas as pd
import numpy as np
import re

# # Titanic Set Revisited with Custom Classes
#
# We'll be introducing some new tools to implement what we did last session. Using these custom classes (regressors, classifiers, cluster-ers, transformers, feature unions and pipelines) can be powerful additions to your tool belt.

# This introduction is modeled after Adam Rogers's titanic_finished-ish.py script we worked through last time.
#
# We start by pulling in the datasets and importing our libraries. Data available at https://www.kaggle.com/c/titanic/data.


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# combining early to apply transformations uniformly
combinedSet = pd.concat([train , test], axis=0)
combinedSet = combinedSet.reset_index(drop=True)

combinedSet.shape

print combinedSet['Survived'].value_counts(dropna=False)

print combinedSet[450:453]

print combinedSet.describe()
## Transformers
# Transformers apply a transformation on the data. We did several 'transformations' on the data last session to prepare it for model fitting. These transformation included:
#
# * filling missing age values;
# * converting Pclass, Embarked, and Deck to indicator variables with pd.get_dummies() (S -> [0, 0, 1])
# * converting gender to a binary variable
# * creating an IsChild indicator variable

# Many other transformers are included in sklearn.preprocessing package (```StandardScaler()```, ```Binarizer()```, ```OneHotEncoder()```, ```Imputer()```, ```LabelEncoder()```, etc.). The feature extraction sklearn module also has many invaluble transformers. Several of which that are great for natural language processing (```TfidfTransformer()```, ```CountVectorizer()```, ```HashingVectorizer()```, etc.).

# These transformers within Sci-kit Learn are classes that have useful methods that can make preparing your data a bit easier. Many tranformers have the ```fit()```, ```transform()```, ```fit_transform()```, and ```inverse_transform()``` methods.

# Last week we had an issue when the test split did not have the same levels of cabin as the train set. One benefit of the transformer model is that it can fit on train data and the same model will be used to map any new data.
#
# If you've ever had to scale your labels and want to get the predicted values back out in the previous scale inverse_transform() helps immensely.
#
# Transformation Benefits:
#
# - better with training with one set and transforming new data
# - can allow for inversing the transformation
# - can allow for more abstraction for common transformations
# - works with pipelines (to be discussed later)

### Some examples:

# Simple scaling:
from sklearn import preprocessing
print pd.DataFrame({'Fare': train.Fare,\
             'ScaledFare': preprocessing.scale(train.Fare)})\
                .head()

# But what if we need to scale new data or  we were predicting fare and need to retrun to the previous scale.

# New data set scaling and reversible scaling:
# instantiate
scaler = preprocessing.StandardScaler()

#fit
scaler.fit(train.Fare.reshape(-1, 1)) # reshape(-1, 1) to get column vector for scaling
testFares =  test.Fare.fillna(0).reshape(-1, 1) # had to fill NA as no NA's in train set, could impute instead

#transform
scaledFare = scaler.transform(testFares)

#inverse transform
inverseScaled = scaler.inverse_transform(scaledFare)

#different from scaling separately, shown if we fit_transform with test alone
scaler2 = preprocessing.StandardScaler()
badScaling = scaler2.fit_transform(testFares)

print pd.DataFrame({'Fare': test.Fare.values,\
             'ScaledFare': scaledFare.T[0],\
             'InverseFare': inverseScaled.T[0],
             'BadScaling': badScaling.T[0]},\
             columns=['Fare', 'ScaledFare','InverseFare', 'BadScaling'])\
                .head()

# Still can be prone to issues if new data has levels/non-numbers not present in train data eg. NaN's for the example above.
# So it is apparent they can add some functionality what about custom transformers?

# ### Custom Transformers
# Can be done simply using ```sklearn.preprocessing.FunctionTransformer()``` or more customization is available from creating a new class inheriting from the base classes: BaseEstimator + what type of estimator you are creating (ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin)  http://scikit-learn.org/stable/modules/classes.html

# Function transformer example:
genderDict = {'male': 1, 'female': 0}
genderFlagger = sk.preprocessing.FunctionTransformer(lambda genderArray: \
                                                   [genderDict[gender] for gender in genderArray],\
                                                   validate=False)
genderFlags= genderFlagger.transform(combinedSet.Sex)
print genderFlags[0:10]

# Not simpler than
# ```python
# passengers["Sex"][passengers["Sex"] == 'male'] = 0
# passengers["Sex"][passengers["Sex"] == 'female'] = 1
# ```
# Pandas makes FunctionTransformer sort of obsolete.

# For more customization we create a new class:
# ``` python
# class NewTransformer(base.BaseEstimator, base.ClassifierMixin):
#     def __init__(self, ...):
#         # initialization code
#
#     def fit(self, X, y=None):
#         # fit the model ...
#         return self
#
#     def transform(self, X):
#         # transformation
#         return new_X
#
#     def fit_transform(self, X, y=None):
#         # fit the model and then transform it
#         return new_X
# ```

# Little more complex for cabin deck.
class cabinLevelsTransformer1(sk.base.BaseEstimator, sk.base.ClassifierMixin):

    # function for extracting deck letter
    def get_deck_letter(self, row):
        # Ignore NaN values
        if not pd.isnull(row["Cabin"]):
            # Get first letter of "Cabin" value
            return str(row["Cabin"])[0]
        # Otherwise return NaN
        return row["Cabin"]

    def transform(self, X):
        # transformation
        newX=X.copy()
        newX["Cabin"] = newX.apply(lambda row: self.get_deck_letter(row), axis=1)
        cabinColumnsDF = pd.get_dummies(newX, columns = ["Cabin"], prefix=['cabin'])
        return cabinColumnsDF

class cabinLevelsTransformer2(sk.base.BaseEstimator, sk.base.ClassifierMixin):
    def __init__(self):
        # initialization code
        self.le = preprocessing.LabelEncoder()
        self.lb = preprocessing.LabelBinarizer()

    # function for extracting deck letter
    def get_deck_letter(self, row):
        # Ignore NaN values
        if not pd.isnull(row["Cabin"]):
            # Get first letter of "Cabin" value
            return str(row["Cabin"])[0]
        # Otherwise return NaN
        return 'NaN'

    def fit(self, X, y = None):
        # fit the model ...
        newX=X.copy()
        newX["Cabin"] = newX.apply(lambda row: self.get_deck_letter(row), axis=1)
        self.le.fit(newX["Cabin"])
        self.lb.fit(self.le.transform(newX["Cabin"]))
        return self

    def transform(self, X):
        # transformation
        newX=X.copy()
        newX["Cabin"] = newX.apply(lambda row: self.get_deck_letter(row), axis=1)
        wtf=self.le.transform(newX["Cabin"])
        cabinColumns = self.lb.transform(wtf)
        cabinColumnsDF = pd.DataFrame(cabinColumns)
        cabinColumnsDF.columns = ['cabin_' + str(cabinNum) for cabinNum in self.le.inverse_transform(cabinColumnsDF.columns)]
        newX = newX.drop('Cabin',1)
        newX = pd.concat([newX.reset_index(drop=True), cabinColumnsDF], axis=1)
        return newX.drop('cabin_NaN', 1)

    def fit_transform(self, X, y=None):
        # fit the model and then transform it
        self.fit(X)
        return  self.transform(X)

clTrans1=cabinLevelsTransformer1()
clTrans2=cabinLevelsTransformer2()

print clTrans1.transform(combinedSet).head(2)

# The more complex cabinLevelsTransformer2 allows for transforming on a set that may not include all the cabin levels. Note: Label Encoder seems to not play well with transforming NaN's so NaN's should be labelled say 'NaN' in the transformer.

clTrans2.fit(train)

print test.loc[12:14,:]
clTrans2.transform(test.loc[12:14,:])

# Keep in mind we don't want to drop any additional columns as NaN's are represented as 0's across cabins.
# Our transformer looks like it'd be pretty similar for any categorical columns. We can make a more generic version.

class genericLevelsToDummiesTransformer(sk.base.BaseEstimator, sk.base.ClassifierMixin):
    def __init__(self, columns, printFlag = False):
        # initialization code
        self.columns = columns
        self.leDict = {}
        self.lbDict = {}
        self.printFlag = printFlag
        self.newColumnNames = {}
        for column in columns:
            # unique transformers for each column
            self.leDict[column] = preprocessing.LabelEncoder()
            self.lbDict[column] = preprocessing.LabelBinarizer()

    # function for extracting deck letter
    def get_deck_letter(self, row):
        # Ignore NaN values
        if not pd.isnull(row["Cabin"]):
            # Get first letter of "Cabin" value
            return str(row["Cabin"])[0]
        # Otherwise return NaN
        return 'NaN'

    def get_title(self, row):
        if not pd.isnull(row["Name"]):
            reResult = re.findall(r'Mr\.|Mrs\.|Rev\.|Miss\.|Jr|Dr\.|Rev.|Master', row["Name"])
            if len(reResult)<1:
                return 'NaN'
            else:
                return reResult[0]
        return 'NaN'

    def fit(self, X, y = None):
        # fit the model ...
        newX=X.copy()
        for column in self.columns:
            if column == 'Cabin':
                newX[column] = newX.apply(lambda row: self.get_deck_letter(row), axis=1)
            elif column == 'Title':
                newX[column] = newX.apply(lambda row: self.get_title(row), axis=1)
            self.leDict[column].fit(newX[column])
            self.lbDict[column].fit(self.leDict[column].transform(newX[column]))
        return self

    def transform(self, X):
        # transformation
        newX=X.copy()
        if self.printFlag: print newX
        for column in self.columns:

            if column == 'Cabin':
                newX["Cabin"] = newX.apply(lambda row: self.get_deck_letter(row), axis=1)
            elif column == 'Title':
                newX[column] = newX.apply(lambda row: self.get_title(row), axis=1)

            # convert to numeric
            newX[column] = self.leDict[column].transform(newX[column])
            if self.printFlag: print newX[column]

            # make dummies
            newColumnsDF = pd.DataFrame(self.lbDict[column].transform(newX[column]))

            # rename dummies to original category levels
            self.newColumnNames[column]=[column+ '_' + str(index) \
                                         for index in self.leDict[column].inverse_transform(newColumnsDF.columns)]
            newColumnsDF.columns = self.newColumnNames[column]
            if self.printFlag: print newColumnsDF

            newX = newX.drop(column,1)
            newX = pd.concat([newX.reset_index(drop=True), newColumnsDF], axis=1)
        return newX

    def fit_transform(self, X, y=None):
        # fit the model and then transform it
        self.fit(X)
        return  self.transform(X)

    def inverse_transform(self, X):
        newX=X.copy()
        for column in self.columns:
            if self.printFlag: print newX.loc[:,self.newColumnNames[column]].values
            invNumColumn = self.lbDict[column].inverse_transform(newX.loc[:,self.newColumnNames[column]].values)
            if self.printFlag: print invNumColumn
            invColumnDF = pd.Series(self.leDict[column].inverse_transform(invNumColumn), name=column)
            if self.printFlag: print invColumnDF
            for dropColumn in self.newColumnNames[column]:
                if dropColumn in newX.columns:
                    newX = newX.drop(dropColumn,1)
            newX = pd.concat([newX.reset_index(drop=True), invColumnDF], axis=1)
        return newX

dummyTransformer=genericLevelsToDummiesTransformer(['Cabin','Sex', 'Pclass','Embarked'], printFlag=False)
dummyTransformer.fit(combinedSet)
print dummyTransformer.transform(test).head(3)
print dummyTransformer.inverse_transform(dummyTransformer.transform(train).head(3))
print combinedSet.head(3)

# The nice thing is now if we realize a new categorical feature we'd like to add it's a two line command after creating the categorical column.
#
# Like titles in the name...

import operator
words = [word for this_name in combinedSet.Name for word in this_name.split(' ')]
d = {}
for word in words:
    if word in d.keys():
        d[word] += 1
    else:
        d[word] = 1

sorted_words = sorted(d.items(), key = operator.itemgetter(1), reverse = True)
print sorted_words[1:10]

def get_title(row):
    if not pd.isnull(row["Name"]):
        reResult = re.findall(r'Mr\.|Mrs\.|Rev\.|Miss\.|Jr|Dr\.|Rev.|Master', row["Name"])
        if len(reResult)<1:
            return 'NaN'
        else:
            return reResult[0]

dummyTransformer=genericLevelsToDummiesTransformer(['Cabin','Sex', 'Pclass','Embarked', 'Title'], printFlag=False)
print dummyTransformer.fit_transform(combinedSet).head(3)

print dummyTransformer.fit_transform(combinedSet).columns

# ## Estimators
# We are familiar with estimators like LogisticRegression, NearestNeighbors, and DecisionTreeClassifier.
#
# We instantiate an estimator, fit, and predict.

from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

print combinedSet.loc[1:10,['Pclass', 'Age', 'Fare', 'Survived']]

X_tree = combinedSet.loc[:,['Age', 'Fare']]\
        .fillna(0)
y_tree = combinedSet['Survived']\
        .fillna(0)

X_tree_train, X_tree_test, y_tree_train, y_tree_test = \
    cross_validation.train_test_split(X_tree, y_tree, random_state=13)

treeClf= tree.DecisionTreeClassifier()
treeClf.fit(X_tree_train, y_tree_train)

tree_predictions = treeClf.predict(X_tree_test)

print 'Mean Accuracy Score: ', treeClf.score(X_tree_test, y_tree_test)
print 'Confusion Matrix: \n'
print pd.DataFrame(confusion_matrix(tree_predictions, y_tree_test))

# Let's create a custom estimator based on the majority survival rate grouped by passenger class, e.g. if most the people in 1st class survived, estimate any test observation from first class survived.

# You will want to include the fit, predict, and score methods:
# ``` python
# class Estimator(base.BaseEstimator, base.ClassifierMixin):
#   def __init__(self, ...):
#   # initialization code
#
#   def fit(self, X, y):
#   # fit the model ...
#     return self
#
#   def predict(self, X):
#     return # prediction
#
#   def score(self, X, y):
#     return # custom score implementation
# ```
### Example to show customization of inputs compared to base estimators:

class PClassEstDFonly(sk.base.BaseEstimator, sk.base.ClassifierMixin):
    def __init__(self):
        # initialization code
        self.modelDF=pd.DataFrame()

    def fit(self, train_DF):
        #fit the model to the majority vote
        self.modelDF=train_DF.loc[:,['Pclass', 'Survived']]\
                        .groupby('Pclass')\
                        .mean()\
                        .round()\
                        .astype(int)
        return self

    def predict(self, test_DF):
        return self.modelDF.loc[test_DF['Pclass'], 'Survived'].values

    def score(self, X, y):
        # custom score implementation
        #F1 score : 2 * precision * recall/(precision + recall)
        predictions = self.predict(X)

        # true positives
        tp = sum(predictions * y) * 1.0
        # false positives
        fp = sum((1-predictions) * y) * 1.0
        # false negatives
        fn = sum(predictions * (1-y)) * 1.0

        precision =  tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall/(precision + recall)

pClassClfDFonly= PClassEstDFonly()
pClassClfDFonly.fit(train[1:700])

print pClassClfDFonly.score(train[701:], train.Survived[701:])

### Example to follow fit(X, y), predict(X) pattern:
class PClassEst2(sk.base.BaseEstimator, sk.base.ClassifierMixin):
    def __init__(self):
        # initialization code
        self.modelDF=pd.DataFrame()

    def fit(self, train_DF, train_labels):
        #fit the model to the

        self.modelDF=train_DF.loc[:,['Pclass', 'Survived']]\
                        .groupby('Pclass')\
                        .mean()\
                        .round()

        return self

    def predict(self, test_DF):
        return self.modelDF.loc[test_DF['Pclass'], 'Survived']

    def score(self, X, y):
        # custom score implementation
        # F1 score : 2 * precision * recall/(precision + recall)
        predictions = self.predict(X)
        # let's use scikit learn's implementation
        return sk.metrics.f1_score(y, predictions)

pClassClfDFonly2= PClassEst2()
pClassClfDFonly2.fit(train[1:700], train.Survived[1:700])

print pClassClfDFonly.score(train[701:], train.Survived[701:])

# ## Pipelines
# Pipelines allow for a claen implementation of a dataset -> transform/manipulate -> predict -> score -> iterate model.

X_fit, X_validation, y_fit, y_validation = \
    cross_validation.train_test_split(train.drop('Survived', axis=1), train.Survived, random_state=13)
# Starting with a simple example transform gender tags and predict using a decsion tree.
# First using nested functions:

dummyTransformer=genericLevelsToDummiesTransformer(['Cabin','Sex', 'Pclass','Embarked'], printFlag=False)

dummyTransformer.fit(train)
dummyTransformer.transform(test).head(3)

def dropColumns(fullColumnDF):
    reducedColumns = fullColumnDF.drop(['PassengerId', 'Name', "Ticket"], axis=1)

    # fill NA's while we're at it
    return reducedColumns.fillna(0)

dropColumns(train).head()
columnDropper = preprocessing.FunctionTransformer(dropColumns, validate=False)
print columnDropper.transform(train).head()

treeClf = tree.DecisionTreeClassifier(random_state=42)
withDummies=dummyTransformer.transform(X_fit)
withDummiesExtraColumnsDropped=columnDropper.transform(withDummies)
print withDummiesExtraColumnsDropped.head()

treeClf.fit(withDummiesExtraColumnsDropped, y_fit)

withDummiesTest=dummyTransformer.transform(X_validation)
withDummiesExtraColumnsDroppedTest=columnDropper.transform(withDummiesTest)

print treeClf.predict(withDummiesExtraColumnsDroppedTest)[0:5]
print treeClf.score(withDummiesExtraColumnsDroppedTest, y_validation)
# Let's change a few of the parameters from the default and see how it affects the score.

treeClf2 = tree.DecisionTreeClassifier(min_samples_leaf=2, max_features=4, min_samples_split=15, random_state=42)
treeClf2.fit(withDummiesExtraColumnsDropped, y_fit)

print treeClf2.score(withDummiesExtraColumnsDroppedTest, y_validation)

# Sweeeeet! We just got more accurate. Is this the best we can do? We should try a bunch of values

treeClf3 = tree.DecisionTreeClassifier(min_samples_leaf=5, max_features=10, min_samples_split=10, random_state=42)
treeClf3.fit(withDummiesExtraColumnsDropped, y_fit)

print treeClf3.score(withDummiesExtraColumnsDroppedTest, y_validation)

# As a pipeline:

from sklearn import pipeline

dummifier = genericLevelsToDummiesTransformer(['Cabin','Sex', 'Pclass','Embarked'], printFlag=False)
dropifier = preprocessing.FunctionTransformer(dropColumns, validate=False)
treeClfPipe = tree.DecisionTreeClassifier(random_state=42)

dummyTreePipeline = pipeline.Pipeline([('dummyMaker', dummifier),
                                        ('columnDropper', dropifier),
                                        ('treeClassifer', treeClfPipe)])

print dummyTreePipeline

dummyTreePipeline.fit(X_fit, y_fit)

print dummyTreePipeline.predict(X_validation)[1:10]
print dummyTreePipeline.score(X_validation,y_validation)

# Changing the parameters now in pipeline version.
dummyTreePipeline.set_params(treeClassifer__min_samples_leaf=5,
                             treeClassifer__max_features=10,
                             treeClassifer__min_samples_split=10)

dummyTreePipeline.fit(X_fit, y_fit)
print dummyTreePipeline.score(X_validation,y_validation)

# Well this is sort of helpful.
# We've seen better parameters can really improve our model. Can we semi automate it? That would really make this powerful stuff.

# ## Cross Validation
# So far we have considered three sets:
#  - fit set to fit the model on (75% of the full train set in the pipeline notebook)
#  - validation set to test the model on with known labels (25% of the full train set in the pipeline notebook)
#  - test set to predict our unknown labels

from sklearn import pipeline
from sklearn import preprocessing
from sklearn import tree
from scikitDemoHelpers import genericLevelsToDummiesTransformer
from scikitDemoHelpers import dropColumns

dummyTransformer=genericLevelsToDummiesTransformer(['Cabin','Sex', 'Pclass','Embarked', 'Title'], printFlag=False)
dummyTransformer.fit_transform(combinedSet)

dropifier = preprocessing.FunctionTransformer(dropColumns, validate=False)
treeClfPipe = tree.DecisionTreeClassifier(random_state=42)

dummyTreePipeline = pipeline.Pipeline([('columnDropper', dropifier),
                                        ('treeClassifer', treeClfPipe)])
dummyTreePipeline.set_params(treeClassifer__min_samples_leaf=5,
                             treeClassifer__max_features=10,
                             treeClassifer__min_samples_split=10)

from sklearn import cross_validation
X_fit, X_validation, y_fit, y_validation = \
    cross_validation.train_test_split(train.drop('Survived', axis=1),
                                      train.Survived, test_size=0.25, random_state=42)

# If we want to consider more than the single train-validation pair, we can cut up the train set into multiple blocks and rotate which one to validate against. This is called K-folds, where K is the number of "folds" we want in the train set.                                      For example with 100 obeservations and k=4:
#  - fold 1: 1-25
#  - fold 2: 26-50
#  - fold 3: 51-75
#  - fold 4: 76-100
#
# We would then crossvalidate 4 times:
#  - fit on fold 2,3,4; validate with fold 1
#  - fit on fold 1,3,4; validate with fold 2
#  - fit on fold 1,2,4; validate with fold 3
#  - fit on fold 1,2,3; validate with fold 4
#
# We would then end up with 4 scores who's average would be better at estimating the model's ability to generalize.
# A simple example below shows that training and testing our model on different sets even of the same size can produce better or worse scores.

from sklearn import metrics
from sklearn import cross_validation

scores =  cross_validation.cross_val_score(dummyTreePipeline,
                                           dummyTransformer.transform(train.drop('Survived',
                                                      axis=1)),
                                           train.Survived,
                                          cv = 5,
                                          n_jobs=-1)
print scores

# If we want to create the folds similar to [```sklearn.crossvalidation.train_test_split()```](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html#sklearn.cross_validation.train_test_split) before we can use [```sklearn.crossvalidation.KFold()```](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold) or [```sklearn.crossvalidation.StratifiedKFold()```](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html#sklearn.cross_validation.StratifiedKFold). Stratified K Folds ensures there is an equal ratio of labels across the folds.

skf = cross_validation.StratifiedKFold(train.Survived, n_folds=5, shuffle=True)
for fit_indices, validate_indices in skf:
    dummyTreePipeline.fit(dummyTransformer.transform(train.drop('Survived', axis=1)).loc[fit_indices,:],
                          train.Survived[fit_indices])
    print dummyTreePipeline.score(dummyTransformer.transform(train.drop('Survived', axis=1)).loc[validate_indices,:], train.Survived[validate_indices])

# Now that we understand the concept of k-folds let's look at gridsearchCV.

# Grid Search
# [Grid Search](http://scikit-learn.org/stable/modules/grid_search.html#grid-search) allows for a search across the parameter space for a model. For the decision tree example this could be considering all the parameters output when we create the classifer:
#          ```
#          DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#                 max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#                 min_samples_split=2, min_weight_fraction_leaf=0.0,
#                 presort=False, random_state=42, splitter='best')
#                 ```

print dummyTreePipeline.get_params()

# This search can be over a predefined parameter set.
paramGrid = [
    {'treeClassifer__max_features': [1, 5, 12, 25], 'treeClassifer__min_samples_split': [5, 10, 15]}
]
from sklearn import grid_search
dummyGridSearch = grid_search.GridSearchCV(dummyTreePipeline, paramGrid, cv=5)
dummyGridSearch.fit(dummyTransformer.transform(train.drop('Survived', axis=1)), train.Survived)
print dummyGridSearch.best_score_
print dummyGridSearch.grid_scores_
print dummyGridSearch.best_params_
print dummyGridSearch.best_estimator_

# Or we can allow it to randomly roam with [```sklearn.grid_search.RandomizedSearchCV```](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html#sklearn.grid_search.RandomizedSearchCV):
from scipy.stats import randint as sp_randint
paramDists = {
 'treeClassifer__max_features': sp_randint(1,25),
 'treeClassifer__min_samples_split': sp_randint(1, 30),
}

dummyRandomSearch = grid_search.RandomizedSearchCV(dummyTreePipeline, paramDists, cv=5, n_iter=10, random_state=42)
dummyRandomSearch.fit(dummyTransformer.transform(train.drop('Survived', axis=1)), train.Survived)
print dummyRandomSearch.best_score_
print dummyRandomSearch.grid_scores_
print dummyRandomSearch.best_params_
print dummyRandomSearch.best_estimator_
