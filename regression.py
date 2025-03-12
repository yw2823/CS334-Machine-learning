import numpy as np
import matplotlib.pyplot as plt

class BaseRegressor:
    """
    Base class for a generic regressor.
    """
    def __init__(self, num_feats, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10):
        self.W = np.random.randn(num_feats + 1).flatten()
        self.lr = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_feats = num_feats
        self.loss_hist_train = []
        self.loss_hist_val = []
    
    def make_prediction(self, X):
        raise NotImplementedError
    
    def loss_function(self, y_true, y_pred):
        raise NotImplementedError
        
    def calculate_gradient(self, y_true, X):
        raise NotImplementedError
    
    def train_model(self, X_train, y_train, X_val, y_val):
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        prev_update_size, iteration = 1, 1

        while prev_update_size > self.tol and iteration < self.max_iter:
            shuffled_indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]
            num_batches = (len(X_train) + self.batch_size - 1) // self.batch_size
            X_batches = np.array_split(X_train, num_batches)
            y_batches = np.array_split(y_train, num_batches)
            update_sizes = []
            
            for X_batch, y_batch in zip(X_batches, y_batches):
                y_pred = self.make_prediction(X_batch)
                train_loss = self.loss_function(y_batch, y_pred)
                self.loss_hist_train.append(train_loss)
                grad = self.calculate_gradient(y_batch, X_batch)
                new_W = self.W - self.lr * grad
                update_sizes.append(np.abs(new_W - self.W))
                self.W = new_W
                val_loss = self.loss_function(y_val, self.make_prediction(X_val))
                self.loss_hist_val.append(val_loss)
            
            prev_update_size = np.mean(update_sizes)
            iteration += 1
    
    def plot_loss_history(self):
        assert self.loss_hist_train, "Training must be run before plotting loss history."
        fig, axs = plt.subplots(2, figsize=(8, 8))
        fig.suptitle('Loss History')
        axs[0].plot(self.loss_hist_train, label='Training Loss')
        axs[0].set_title('Training')
        axs[1].plot(self.loss_hist_val, label='Validation Loss')
        axs[1].set_title('Validation')
        plt.xlabel('Steps')
        fig.tight_layout()
        plt.show()

    def reset_model(self):
        self.W = np.random.randn(self.num_feats + 1).flatten()
        self.loss_hist_train = []
        self.loss_hist_val = []

class LogisticRegressor(BaseRegressor):
    """
    Logistic regression model inheriting from BaseRegressor.
    """
    def __init__(self, num_feats, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10):
        super().__init__(num_feats, learning_rate, tol, max_iter, batch_size)
    
    def make_prediction(self, X) -> np.ndarray:
        return 1 / (1 + np.exp(-np.dot(X, self.W)))
        
    def loss_function(self, y_true, y_pred) -> float:
        return np.mean(-y_true * np.log(y_pred + 1e-9) - (1 - y_true) * np.log(1 - y_pred + 1e-9))
        
    def calculate_gradient(self, y_true, X) -> np.ndarray:
        return np.dot(X.T, self.make_prediction(X) - y_true) / X.shape[0]
