import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

class ArxivClassifier:
    def __init__(self, preprocessor, random_state=42):
        """Initialize classifier with preprocessor pipeline"""
        self.preprocessor = preprocessor
        self.random_state = random_state
        self.model = None
        self.pipeline = None
        self.classes_ = None
        
    def create_model(self, model_type='gb'):
        """Create the specified model with optimized hyperparameters"""
        if model_type.lower() == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=self.random_state
            )
        else:
            raise ValueError("Only Gradient Boosting ('gb') is supported")
            
        # Create pipeline with preprocessor and model
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", self.model)
            ]
        )
        
        return self.pipeline
        
    def get_param_grid(self, model_type='rf'):
        """Get parameter grid for grid search"""
        if model_type.lower() == 'rf':
            return {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2']
            }
        elif model_type.lower() == 'dt':
            return {
                'classifier__max_depth': [10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__criterion': ['gini', 'entropy']
            }
        elif model_type.lower() == 'gb':
            return {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5],
                'classifier__min_samples_split': [5, 10],
                'classifier__min_samples_leaf': [2, 5],
                'classifier__subsample': [0.8, 1.0]
            }
        
    def train(self, X_train, y_train, do_grid_search=False, model_type='rf'):
        """Train the model with optional grid search"""
        if self.pipeline is None:
            self.create_model(model_type)
            
        if do_grid_search:
            print(f"Performing grid search for {model_type}...")
            param_grid = self.get_param_grid(model_type)
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=3,
                n_jobs=-1,
                scoring='f1_macro',
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print("Best parameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)
            
            self.pipeline = grid_search.best_estimator_
        else:
            self.pipeline.fit(X_train, y_train)
            
        self.classes_ = self.pipeline.classes_
        
    def predict(self, X):
        """Make predictions on new data"""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first")
        return self.pipeline.predict(X)
        
    def predict_proba(self, X):
        """Get probability estimates"""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first")
        return self.pipeline.predict_proba(X)
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first")
            
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Get classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.classes_,
            output_dict=True
        )
        
        return report
        
    def cross_validate(self, X_train, y_train, cv=3, metrics=['accuracy','f1_macro']):
        """Perform cross validation"""
        if self.pipeline is None:
            self.create_model()
            
        # Perform cross validation
        cv_results = cross_validate(
            self.pipeline, X_train, y_train,
            cv=cv,
            return_train_score=True,
            scoring=metrics,
            n_jobs=-1
        )
        
        return cv_results
