import pandas as pd
import numpy as np
import sys
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.label_map = {"Yes": 1, "No": 0}
        self.reverse_map = {1: "Yes", 0: "No"}
        self.tree_ = None

    def train(self, X, y):
        y = np.array([self.label_map[label] for label in y])
        self.tree_ = self._build_tree(X, y)

    def predict(self, X):
        return [self.reverse_map[self._predict(inputs, self.tree_)] for inputs in X]

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]
        
        best_feat, best_value = self._best_split(X, y)
        if best_feat is None:
            return Counter(y).most_common(1)[0][0]
        
        left_mask = X[:, best_feat] <= best_value
        right_mask = ~left_mask
        return {
            'feature_index': best_feat,
            'threshold': best_value,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _best_split(self, X, y):
        best_gini, best_feat, best_value = float("inf"), None, None
        for feat in range(X.shape[1]):
            for value in np.unique(X[:, feat]):
                gini = self._gini_index(X[:, feat], y, value)
                if gini < best_gini:
                    best_gini, best_feat, best_value = gini, feat, value
        return best_feat, best_value

    def _gini_index(self, feature, labels, threshold):
        def gini(y):
            probs = np.bincount(y) / len(y) if len(y) > 0 else [0]
            return 1.0 - np.sum(probs ** 2)
        left, right = labels[feature <= threshold], labels[feature > threshold]
        return (len(left) * gini(left) + len(right) * gini(right)) / len(labels)

    def _predict(self, inputs, tree):
        while isinstance(tree, dict):
            tree = tree['left'] if inputs[tree['feature_index']] <= tree['threshold'] else tree['right']
        return tree

def k_fold_cross_validation(data, k=10):
    fold_size = len(data) // k
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}
    
    for i in range(k):
        test_data = data.iloc[i * fold_size:(i + 1) * fold_size]
        train_data = pd.concat([data.iloc[:i * fold_size], data.iloc[(i + 1) * fold_size:]])
        
        X_train, y_train = train_data.drop(['person ID', 'Has heart disease?'], axis=1).values, train_data['Has heart disease?'].values
        X_test, y_test = test_data.drop(['person ID', 'Has heart disease?'], axis=1).values, test_data['Has heart disease?'].values
        
        model = DecisionTree(max_depth=5)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        actual, predicted = np.array(y_test), np.array(predictions)
        tp = np.sum((predicted == "Yes") & (actual == "Yes"))
        fp = np.sum((predicted == "Yes") & (actual == "No"))
        fn = np.sum((predicted == "No") & (actual == "Yes"))
        tn = np.sum((predicted == "No") & (actual == "No"))
        
        metrics['accuracy'].append(np.mean(actual == predicted))
        metrics['precision'].append(tp / (tp + fp) if tp + fp > 0 else 0)
        metrics['recall'].append(tp / (tp + fn) if tp + fn > 0 else 0)
        metrics['f1'].append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0)
        metrics['specificity'].append(tn / (tn + fp) if tn + fp > 0 else 0)
    
    print("Average Metrics:")
    for key, values in metrics.items():
        print(f"{key.capitalize()}: {np.mean(values):.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python DecisionTree.py <input_file>')
        sys.exit(1)
    
    data = pd.read_csv(sys.argv[1])
    k_fold_cross_validation(data, k=10)
