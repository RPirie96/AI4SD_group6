from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=3)
results = cross_val_score(clf ,X_train_transformed,y_train_translated, scoring="roc_auc")
print(results)