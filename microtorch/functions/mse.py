def mean_squared_error(y_pred, y_true):
    """Mean Squared Error"""
    return ((y_pred - y_true) ** 2).mean()