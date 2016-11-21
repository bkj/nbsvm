import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2

train = pd.read_csv('./ft-train.txt', sep='\t', header=None)
test = pd.read_csv('./ft-test.txt', sep='\t', header=None)

X_train, y_train = train[1], train[0]
X_test, y_test = test[1], test[0]

Cs = np.hstack([np.arange(0.01, 0.5, 0.01)])
lr = LogisticRegressionCV(Cs=Cs, n_jobs=5, class_weight='balanced')
tf = TfidfVectorizer(min_df=10, max_df=0.95, ngram_range=(1, 2))

Xv_train = tf.fit_transform(X_train)
lr = lr.fit(Xv_train, y_train)

Xv_test = tf.transform(X_test)
test_preds = lr.predict_proba(Xv_test)[:,1]

roc_auc_score(y_test == '__label__weary', test_preds)
((test_preds > 0.5) == (y_test == '__label__weary')).mean()
