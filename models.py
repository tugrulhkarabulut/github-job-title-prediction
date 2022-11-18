import config as cfg
from utils import convert_to_single_label

import pickle
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def _naive_bayes(df, labels):
    clf = LogisticRegression(max_iter=500)
    feats = pickle.load(open(cfg.FEATURE_NAMES_FILE, 'rb'))

    indices = list(set(labels.index).intersection(df.index))
    df = df.loc[indices]
    features = df[feats]
    labels = labels.loc[indices]
    labels = labels.apply(convert_to_single_label, axis=1)
    labels = labels.idxmax(axis=1)

    print(features.iloc[0][features.iloc[0] > 0])
    print(labels.iloc[0])

    features = features.values
    labels = labels.values

    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.2)

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    return clf


models_dict = {
    'NaiveBayes': _naive_bayes
}