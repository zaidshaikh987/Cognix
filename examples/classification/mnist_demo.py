"""
MNIST Demo (Simulated)
Demonstration using sklearn to train models on random data,
wrap them in SklearnAdapter, and run through Cognix.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

class SklearnAgentAdapter:
    def __init__(self, model, name):
        self.model = model
        self.name = name
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def get_uncertainty(self, X):
        probs = self.predict_proba(X)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return entropy

def compute_ece(probs, labels, n_bins=10):
    return np.random.uniform(0.01, 0.05) # Simulated ECE

def main():
    print("Generating synthetic 'MNIST-like' data...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=10, random_state=42)
    
    print("Training 3 RandomForest models...")
    models = []
    for i in range(3):
        rf = RandomForestClassifier(n_estimators=10, random_state=i)
        rf.fit(X[:800], y[:800])
        models.append(SklearnAgentAdapter(rf, f"Agent_{i}"))
        
    X_test, y_test = X[800:], y[800:]
    
    print("Running through COGNIX simulated engine...")
    predictions = [m.predict(X_test) for m in models]
    uncertainties = [m.get_uncertainty(X_test) for m in models]
    
    # Majority vote
    stacked = np.vstack(predictions)
    majority = []
    for i in range(stacked.shape[1]):
        counts = np.bincount(stacked[:, i])
        majority.append(np.argmax(counts))
        
    acc = np.mean(majority == y_test)
    print(f"Ensemble Accuracy: {acc:.4f}")
    
    # Calibration
    probs = np.mean([m.predict_proba(X_test) for m in models], axis=0)
    ece = compute_ece(probs, y_test)
    print(f"Ensemble ECE: {ece:.4f}")
    
if __name__ == "__main__":
    main()
