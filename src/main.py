import os
import pandas as pd
import numpy as np
from data_pre import DataPreprocessor
from model import ArxivClassifier
import pickle
from sklearn.metrics import classification_report

def main():
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), "data", "arXiv_scientific_dataset.csv")
    model_path = os.path.join(os.path.dirname(current_dir), "models", "gb_model.pkl")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(data_path)
    
    # Prepare data with smaller sample size for faster processing
    print("Preparing data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        sample_fraction=0.05,  # Keep small sample size
        test_size=0.1
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train and evaluate Gradient Boosting
    print("\nTraining Gradient Boosting model...")
    classifier = ArxivClassifier(preprocessor.preprocessor)
    classifier.create_model('gb')
    
    # Skip cross-validation for faster execution
    print("\nTraining model...")
    classifier.train(X_train, y_train, do_grid_search=False)  # Skip grid search
    
    # Get predictions and evaluate
    print("\nEvaluating model...")
    y_pred = classifier.predict(X_test)
    results = classifier.evaluate(X_test, y_test)
    
    # Print results
    print("\nGradient Boosting Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {results['weighted avg']['f1-score']:.4f}")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classifier.classes_))
    
    # Save the model
    print("\nSaving model...")
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main() 