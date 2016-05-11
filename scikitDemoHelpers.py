import sklearn as sk
import pandas as pd
import numpy as np
import re

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
            self.leDict[column] = sk.preprocessing.LabelEncoder()
            self.lbDict[column] = sk.preprocessing.LabelBinarizer()

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
