import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split


def split_train_and_set(df):
    train, test = train_test_split(df, test_size=0.20, random_state=42)
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    return train_df, test_df


def train_model(x_train, y_train):
    # clf = LogisticRegression(C=10, random_state=0)
    # clf = SGDClassifier(loss='hinge', penalty=None, alpha=1e-3, random_state=42)
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    return clf


def use_model(clf, x_test, y_test):
    y_predict = clf.predict(x_test)

    # display accuracy, percision and recall
    eval_classifier = evaluate_prediction(y_predict, y_test)

    # display confusion matrix and normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, x_test, y_test, normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    print(eval_classifier)


def get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class):
    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold


def evaluate_prediction(y_predict, y_test, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    percision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    if y_proba is not None:
        # compute ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        # Compute Area Under the ROC from prediction scores
        roc_auc = roc_auc_score(y_test, y_proba)

        # compute best threshold
        _, best_threshold = get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, tp + fn, tn + fp)

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        return {'percision': percision, 'recall': recall, 'accuracy': accuracy, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'best_threshold': best_threshold}

    return {'percision': percision, 'recall': recall, 'accuracy': accuracy}