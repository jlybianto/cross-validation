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

