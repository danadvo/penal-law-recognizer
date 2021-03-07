import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split


def split_train_and_set(df):
    train, test = train_test_split(df, test_size=0.20, random_state=42)
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    return train_df, test_df


def train_model(x_train, y_train):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
    # %% LogisticRegressionClassifier
    lr_clf = LogisticRegression(penalty='l2', C=10)
    lr_clf.fit(x_train, y_train)
    return lr_clf


def evaluate_model(lr, data, labels):
    pass


def use_model(clf, x_test, y_test):
    y_proba = clf.predict_proba(x_test)[:, 1]
    fpr, tpr, roc_auc = auc(y_test, y_proba)
    y_predict = list(map(lambda x: 1 if x > 0.513 else 0, y_proba))
    eval_classifier = evaluate_prediction(y_proba, y_predict, y_test)
    print(eval_classifier)


def get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class):
    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold


def evaluate_prediction(y_proba, y_predict, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    percision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    # compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    # Compute Area Under the ROC from prediction scores
    roc_auc = roc_auc_score(y_test, y_proba)

    # compute best threshold
    _, best_threshold = get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, tp + fn, tn + fp)
    # print(f"best_threshold = {best_threshold}")
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    # print([success_rate, percision, recall, accuracy])
    return {'percision': percision, 'recall': recall, 'accuracy': accuracy, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'best_threshold': best_threshold}


def auc(y_test, y_predict_proba):
    # compute ROC
    fpr, tpr, _ = roc_curve(y_test, y_predict_proba)
    # Compute Area Under the ROC from prediction scores
    roc_auc = roc_auc_score(y_test, y_predict_proba)
    print(f"roc_auc = {roc_auc}")
    return fpr, tpr, roc_auc