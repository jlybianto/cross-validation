# ----------------
# IMPORT PACKAGES
# ----------------

import pandas as pd
import numpy as np
from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split, cross_val_score

# ----------------
# OBTAIN DATA
# ----------------

iris = datasets.load_iris()

# ----------------
# PROFILE DATA
# ----------------

data = pd.DataFrame(iris.data, columns = iris.feature_names)
data["Species"] = iris.target
print(data)
print("")

# ----------------
# MODEL DATA (WITHOUT SPLITTING INTO TRAINING / TESTING)
# ----------------

# Dataset not split into training and testing
print("Model Data without Splitting: ")
print("Data points in Data Set: " + str(len(data)))
svc = svm.SVC(kernel="linear")
x = iris.data
y = iris.target
svc.fit(x, y)

predict = svc.predict(x)
expect = y
match = 0
for i in predict:
	if i == expect[i]:
		match += 1

score = round((float(match) / len(iris.target)) * 100, 1)
print("Number of Matches: " + str(match))
print("Score: " + str(score) + "%")
print("")

# ----------------
# MODEL DATA (WITH SPLITTING INTO TRAINING / TESTING)
# ----------------

# Split dataset into training and testing (60-40 ratio)
print("Model Data with Splitting: ")
train, test = train_test_split(data, test_size=0.4)
print("Data points in Training Set: " + str(len(train)))
print("Data points in Test Set: " + str(len(test)))

svc_split = svm.SVC(kernel="linear")
x_split = train.ix[:, 0:4]
y_split = train.ix[:, 4]
svc_split.fit(x_split, y_split)

predict_split = svc.predict(test.ix[:, 0:4])
expect_split = np.array(test.ix[:, 4])
match_split = 0
for i in predict_split:
	if i == expect_split[i]:
		match_split += 1

score_split = round((float(match_split) / len(test)) * 100, 1)
print("Number of Matches: " + str(match_split))
print("Score: " + str(score_split) + "%")
print("")

# ----------------
# MODEL DATA (CROSS-VALIDATION)
# ----------------

print("Model Data with Cross-Validation: ")
k = int(raw_input("Insert integer number of cross-validations (k): "))
score_cross = cross_val_score(svc, x, y, cv=k)
print("Accuracy: %0.2f +/- %0.2f" % (score_cross.mean(), score_cross.std() * 2))