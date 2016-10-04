import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from operator import itemgetter
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve

def report(grid_scores, n_top=1):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Modelo en el puesto: {0}".format(i + 1))
        print("Media de la puntuación: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parámetros: {0}".format(score.parameters))
        print(" ")

def fitting(clasificador, param_grid, cv):
    gsclf = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    gsclf.fit(X_train, y_train)
    best_clf = gsclf.best_estimator_
    clf_report = report(gsclf.grid_scores_)
    print(clf_report)
    return best_clf
        
def prediction(best_clf, name):
    yPred = best_clf.predict(X_test)
    yPredPr = best_clf.predict_proba(X_test)[:, 1]
    cfx_mtrx = pd.DataFrame(confusion_matrix(y_test, yPred), 
             index=['Continue', 'Dropout'],
             columns=['pred. Cont', 'pred. DO'])
    
    print(classification_report(y_test, yPred))
    print(cfx_mtrx)
    return yPredPr
    
def rocAuc(y_test, yPred):
    fpr_model, tpr_model, _ = roc_curve(y_test, yPred)
    return fpr_model, tpr_model