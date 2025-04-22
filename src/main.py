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
    
    # Prepare data with larger sample size
    print("Preparing data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        sample_fraction=0.05,  # Increased from 0.15 to 0.3
        test_size=0.1
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Print class distribution
    print("\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True).sort_values(ascending=False))
    
    # Train and evaluate Gradient Boosting
    print("\nTraining Gradient Boosting model...")
    classifier = ArxivClassifier(preprocessor.preprocessor)
    classifier.create_model('gb')
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    cv_results = classifier.cross_validate(X_train, y_train, cv=3)
    print("\nCross-validation results:")
    print(f"Mean train accuracy: {cv_results['train_accuracy'].mean():.4f}")
    print(f"Mean test accuracy: {cv_results['test_accuracy'].mean():.4f}")
    print(f"Mean train F1: {cv_results['train_f1_macro'].mean():.4f}")
    print(f"Mean test F1: {cv_results['test_f1_macro'].mean():.4f}")
    
    # Train the model with grid search
    print("\nTraining model with grid search...")
    classifier.train(X_train, y_train, do_grid_search=True)  # Enabled grid search
    
    # Get predictions and evaluate
    print("\nEvaluating model...")
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    results = classifier.evaluate(X_test, y_test)
    
    # Print detailed results
    print("\nModel Performance Metrics:")
    print("=" * 50)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1 Score: {results['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1 Score: {results['weighted avg']['f1-score']:.4f}")
    print(f"Macro Precision: {results['macro avg']['precision']:.4f}")
    print(f"Macro Recall: {results['macro avg']['recall']:.4f}")
    
    # Print per-class metrics
    print("\nPer-class Performance:")
    print("=" * 50)
    for class_name in classifier.classes_:
        if class_name in results:
            print(f"\n{class_name}:")
            print(f"  Precision: {results[class_name]['precision']:.4f}")
            print(f"  Recall: {results[class_name]['recall']:.4f}")
            print(f"  F1-score: {results[class_name]['f1-score']:.4f}")
            print(f"  Support: {results[class_name]['support']}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("=" * 50)
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(cm)
    
    # Calculate and print misclassification rate
    misclassified = (y_test != y_pred).sum()
    total = len(y_test)
    print(f"\nMisclassification Rate: {misclassified/total:.4f} ({misclassified}/{total})")
    
    # Save the model
    print("\nSaving model...")
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main() 