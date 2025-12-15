from sklearn.metrics import classification_report

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    report = classification_report(y_test, preds, output_dict=True)
    return model, preds, report
