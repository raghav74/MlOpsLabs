def predict_data(model, X):
    """
    Predict the class labels for the input data.
    Args:
        model: Trained model object.
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    y_pred = model.predict(X)
    return y_pred
