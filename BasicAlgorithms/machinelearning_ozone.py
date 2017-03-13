#!/usr/bin/env python
""" Simple Air Quality Machine Learning

Andy Wentland
awentland90@gmail.com

Set Up:

Included is a sample CASTNET AQ and Meteorology input data file that will be used for our model.
I cleaned this up so you will not have to worry about dealing with missing or bad data.

csv_in = "./data/CASTNET_clean_ozone_met_MLready.csv"

The model has 3 indpendent physical variables:
 Temperature, Solar Radiation, and Wind Speed

These three variables were chosen because they are imporant, although not comprehensive, when considering the physical
impact they have on air pollution and particulary ground level ozone formation. I have hand sorted AQI
(Air Quality Index) for this site and temporal period. THIS AQI IS NOT WHAT THE EPA USES.
For simplicity I have generated quartiles based on this observational data sets ozone concentration
and then assigned 4 AQI categories.

O3 Conc	    AQI	        Quartile
0	        Very-low	0
25	        Low	        1
34.7	    Medium	    2
44.9225	    High	    3

Usage:
$ python machinelearning_ozone.py

This example borrows from the framework Jason Brownlee introduces for basic machine learning
http://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

"""
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib
matplotlib.use('TkAgg')  # Needed to display plots in GUI on Mac OS X
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Read in CSV of raw observational data
csv_in = "./data/CASTNET_clean_ozone_met_MLready.csv"
names = ['TEMPERATURE', 'SOLAR_RADIATION', 'WINDSPEED', 'AQI']
dataset = pandas.read_csv(csv_in, names=names)

# Do some quick summaries of the raw data set so we know what we're working with
print("\nHead 20 of Dataset")
print(dataset.head(20))
print("\nDescribe Data")
print(dataset.describe())
print("\nAQI Distribution")
print(dataset.groupby('AQI').size())

# Scatter matrix summary
scatter_matrix(dataset)
plt.savefig("./images/scatter_matrix.png")

# Split data into training and validation
array = dataset.values
X = array[:, 0:3]
Y = array[:, 3]
validation_size = 0.3
seed = 8
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Set options of the algorithms
num_folds = 10
num_instances = len(X_train)
seed = 9
scoring = 'accuracy'


# Spot Check Algorithms
print("Spot Check Models")
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig("./images/algorithm_comparison.png")

# Make predictions on validation dataset
print("\nLinear Discriminant Analysis Now Running...")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print("Linear Discriminant Analysis Results")
print("Accuracy Score")
print(accuracy_score(Y_validation, predictions))
print("Confusion Matrix")
print(confusion_matrix(Y_validation, predictions))
print("Classification Report")
print(classification_report(Y_validation, predictions))
