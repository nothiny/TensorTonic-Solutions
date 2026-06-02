import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    # 初始化参数
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(steps):
        # 前向传播
        linear_model = X @ w + b
        predictions = _sigmoid(linear_model)
        
        # 计算梯度
        error = predictions - y
        dw = (1 / n_samples) * X.T @ error
        db = (1 / n_samples) * np.sum(error)
        
        # 参数更新
        w -= lr * dw
        b -= lr * db
    
    return w, b