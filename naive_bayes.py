from collections import defaultdict, Counter
from math import log

class NaiveBayes:
    """Hybrid Naive Bayes classifier supporting discrete and continuous features."""
    
    def __init__(self, extract_features, use_smoothing=True):
        """Initialize Naive Bayes classifier.
        
        param extract_features: Function to extract discrete and continuous features.
        param use_smoothing: Whether to apply smoothing for probability calculations.
        """
        self.priors = {}
        self.label_counts = Counter()
        self.discrete_features = DiscreteFeatureVectors(use_smoothing)
        self.continuous_features = ContinuousFeatureVectors()
        self._extract_features = extract_features
        self._is_fitted = False
    
    def fit(self, X, y):
        """Train the Naive Bayes classifier.
        
        param X: Training data (m x n), where m is the number of samples and n is the number of features.
        param y: Target labels (m,), corresponding to each training sample.
        """
        for features, label in zip(X, y):
            self.label_counts[label] += 1
            extracted_features = self._extract_features(features)
            for idx, feature in enumerate(extracted_features):
                if feature.is_continuous():
                    self.continuous_features.add(label, idx, feature)
                else:
                    self.discrete_features.add(label, idx, feature)
        
        total_samples = len(y)
        self.priors = {label: count / total_samples for label, count in self.label_counts.items()}
        self.continuous_features.set_mean_variance()
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """Predict labels for a dataset.
        
        param X: Test dataset (m x n), where m is the number of samples.
        return: List of predicted labels.
        """
        self._ensure_fitted()
        return [self._predict_single(sample) for sample in X]
    
    def _predict_single(self, sample):
        """Predict label for a single test sample using log-likelihood maximization.
        
        param sample: Test sample to classify.
        return: Predicted label.
        """
        log_likelihoods = {label: log(prior) for label, prior in self.priors.items()}
        features = self._extract_features(sample)
        
        for label in self.label_counts:
            for idx, feature in enumerate(features):
                probability = self._get_feature_probability(idx, feature, label)
                log_likelihoods[label] += log(probability) if probability > 0 else float('-inf')
        
        return max(log_likelihoods, key=log_likelihoods.get)
    
    def _get_feature_probability(self, idx, feature, label):
        """Compute feature probability given a label.
        
        param idx: Feature index.
        param feature: Feature value.
        param label: Class label.
        return: Probability of the feature given the label.
        """
        return (
            self.continuous_features.probability(label, idx)
            if feature.is_continuous() 
            else self.discrete_features.probability(label, idx, feature, self.label_counts[label])
        )
    
 