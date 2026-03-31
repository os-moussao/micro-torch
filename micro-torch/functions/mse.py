def mse(y_pred, y):
    """Mean Squared Error"""
    return ((y_pred - y) ** 2).mean()