import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

# training data
pp = np.load('../Task3/Task3/perfect_patches.npy')
dp = np.load('../Task3/Task3/defect_patches.npy')

# test data
graphene = np.load('../Task3/Task3/full-stack.npy')

# data set of 50/50 perfect to imperfect images
patches = np.concatenate([pp, dp])  # matrices
gt = np.concatenate([np.zeros(len(pp)), np.ones(len(dp))])  # labels

xy = list(zip(patches, gt))
random.shuffle(xy)  # shuffle the tuples

X_train, X_test, y_train, y_test = train_test_split(
    patches,
    gt,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

nsamples, nx, ny = X_train.shape
X_train_transformed = X_train.reshape((nsamples, nx * ny))

nsamples, nx, ny = X_test.shape
X_test_transformed = X_test.reshape((nsamples, nx * ny))

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_transformed, y_train)

y_pred = sgd_clf.predict(X_test_transformed)
print(np.array(y_pred == y_test))
print('')
print('Percentage correct: ', 100 * np.sum(y_pred == y_test) / len(y_test))

labels = y_test
predictions = y_pred

df = pd.DataFrame(
    np.c_[labels, predictions],
    columns=['true_label', 'prediction']
)
print(df)
