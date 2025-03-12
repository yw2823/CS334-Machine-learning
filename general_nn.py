import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
import copy

class NeuralNetwork:

    def __init__(self, nn_arch: List[Dict[str, Union[int, str]]], lr: float, seed: int,
                 batch_size: int, epochs: int, loss_function: str, patience: int, progress: int):
        self.arch = nn_arch
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        self._patience = patience
        self._progress = progress
        self._param_dict = self._init_params()
    
    def _init_params(self) -> Dict[str, ArrayLike]:
        np.random.seed(self._seed)
        param_dict = {}
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            param_dict[f'W{layer_idx}'] = np.random.randn(layer['output_dim'], layer['input_dim']) * 0.1
            param_dict[f'b{layer_idx}'] = np.random.randn(layer['output_dim'], 1) * 0.1
        return param_dict

    def _single_forward(self, W_curr: ArrayLike, b_curr: ArrayLike, A_prev: ArrayLike, activation: str)\
                        -> Tuple[ArrayLike, ArrayLike]:
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T
        A_curr = self._sigmoid(Z_curr) if activation == "sigmoid" else self._relu(Z_curr)
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        A_curr = X
        cache = {}
        for i, layer in enumerate(self.arch):
            W_curr, b_curr = self._param_dict[f'W{i+1}'], self._param_dict[f'b{i+1}']
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_curr, layer['activation'])
            cache[f'Z{i+1}'], cache[f'A{i+1}'] = Z_curr, A_curr
        return A_curr, cache
    
    def _single_backprop(self, W_curr: ArrayLike, Z_curr: ArrayLike, A_prev: ArrayLike, dA_curr: ArrayLike, activation_curr: str)\
                          -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr) if activation_curr == "sigmoid" else self._relu_backprop(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr.T, A_prev)
        db_curr = np.sum(dZ_curr, axis=0).reshape(W_curr.shape[0], 1)
        dA_prev = np.dot(dZ_curr, W_curr)
        return dA_prev, dW_curr, db_curr
    
    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        dA_curr = self._binary_cross_entropy_backprop(y, y_hat) if self._loss_func == "binary_cross_entropy" \
                  else self._mean_squared_error_backprop(y, y_hat)
        grad_dict = {}
        for i in reversed(range(1, len(self.arch) + 1)):
            dA_curr, dW_curr, db_curr = self._single_backprop(self._param_dict[f'W{i}'], cache[f'Z{i}'],
                                                               cache[f'A{i-1}'] if i > 1 else y_hat, dA_curr, self.arch[i-1]['activation'])
            grad_dict[f'dW{i}'], grad_dict[f'db{i}'] = dW_curr, db_curr
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        for i in range(1, len(self.arch) + 1):
            self._param_dict[f'W{i}'] -= self._lr * grad_dict[f'dW{i}']
            self._param_dict[f'b{i}'] -= self._lr * grad_dict[f'db{i}']
    
    def fit(self, X_train: ArrayLike, y_train: ArrayLike, X_val: ArrayLike, y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        per_epoch_loss_train, per_epoch_loss_val = [], []
        best_val_loss, epochs_no_improve, best_params = np.inf, 0, None
        num_batches = np.ceil(len(y_train) / self._batch_size)
        for e in range(self._epochs):
            shuffle = np.random.permutation(len(y_train))
            X_batches, y_batches = np.array_split(X_train[shuffle], num_batches), np.array_split(y_train[shuffle], num_batches)
            train_losses = []
            for X_batch, y_batch in zip(X_batches, y_batches):
                output, cache = self.forward(X_batch)
                train_losses.append(self._compute_loss(y_batch, output))
                self._update_params(self.backprop(y_batch, output, cache))
            per_epoch_loss_train.append(np.mean(train_losses))
            val_loss = self._compute_loss(y_val, self.predict(X_val))
            per_epoch_loss_val.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss, epochs_no_improve, best_params = val_loss, 0, copy.deepcopy(self._param_dict)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= self._patience:
                print(f"Early stopping at epoch {e + 1}.")
                self._param_dict = best_params
                break
            if e % self._progress == 0:
                print(f"Epoch {e + 1} / {self._epochs} completed.")
        return per_epoch_loss_train, per_epoch_loss_val
    
    def predict(self, X: ArrayLike) -> ArrayLike:
        return self.forward(X)[0]
    
    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        return 1 / (1 + np.exp(-Z))
    
    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return dA * self._sigmoid(Z) * (1 - self._sigmoid(Z))
    
    def _relu(self, Z: ArrayLike) -> ArrayLike:
        return np.maximum(0, Z)
    
    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return dA * (Z > 0).astype(int)
    
    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        y_hat = np.clip(y_hat, 1e-5, 1 - 1e-5)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        return np.mean((y - y_hat) ** 2)
