"""
Advanced Disease Diagnosis Model
Implements Bagging, Boosting, and Hybrid Ensemble Methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,  # Bagging
    GradientBoostingClassifier,  # Boosting
    AdaBoostClassifier,  # Boosting
    BaggingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import pickle
import warnings
warnings.filterwarnings('ignore')


class DiseasePredictor:
    def __init__(self):
        self.models = {}
        self.column_names = None
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and validate dataset"""
        print("\n📂 Loading dataset...")
        data = pd.read_csv(filepath)
        
        # Display dataset info
        print(f"✓ Dataset shape: {data.shape}")
        print(f"✓ Columns: {data.columns.tolist()}")
        print(f"✓ Missing values:\n{data.isnull().sum()}")
        print(f"✓ Data types:\n{data.dtypes}")
        
        # Handle duplicates
        initial_rows = len(data)
        data = data.drop_duplicates()
        print(f"✓ Removed {initial_rows - len(data)} duplicate rows")
        
        # Remove duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]
        
        # Identify target column (disease/prognosis)
        target_cols = [col for col in data.columns.lower() 
                       if any(x in col for x in ['disease', 'prognosis', 'target', 'label'])]
        
        if not target_cols:
            raise ValueError("Could not find target column. Check dataset columns.")
        
        target_col = target_cols[0]
        print(f"✓ Target column: {target_col}")
        
        # Prepare features and target
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        self.column_names = X.columns.tolist()
        
        # Encode target variable if categorical
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            print(f"✓ Encoded {len(self.label_encoder.classes_)} disease classes")
        
        # Data statistics
        print(f"\n📊 Data Statistics:")
        print(f"✓ Features: {X.shape[1]}")
        print(f"✓ Samples: {X.shape[0]}")
        print(f"✓ Target classes: {np.unique(y)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print(f"\n✓ Training set: {self.X_train.shape[0]} samples")
        print(f"✓ Testing set: {self.X_test.shape[0]} samples")
    
    def train_decision_tree(self):
        """Baseline Decision Tree Classifier"""
        print("\n🌳 Training Decision Tree...")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = model
        return model
    
    def train_bagging_models(self):
        """Train Bagging-based ensemble models"""
        print("\n🎒 Training Bagging Models...")
        
        # Random Forest (Bagging of Decision Trees)
        print("   → Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # Bagging Classifier
        print("   → Bagging Classifier...")
        bagging = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        bagging.fit(self.X_train, self.y_train)
        self.models['Bagging'] = bagging
    
    def train_boosting_models(self):
        """Train Boosting-based ensemble models"""
        print("\n🚀 Training Boosting Models...")
        
        # AdaBoost Classifier
        print("   → AdaBoost...")
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        )
        ada.fit(self.X_train, self.y_train)
        self.models['AdaBoost'] = ada
        
        # Gradient Boosting Classifier
        print("   → Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
    
    def train_voting_ensemble(self):
        """Train Voting Classifier (Hybrid Ensemble)"""
        print("\n🗳️ Training Voting Ensemble...")
        
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('ada', AdaBoostClassifier(n_estimators=30, random_state=42))
            ],
            voting='soft'  # Use probability predictions
        )
        voting_clf.fit(self.X_train, self.y_train)
        self.models['Voting Ensemble'] = voting_clf
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*80)
        print("📈 MODEL PERFORMANCE EVALUATION")
        print("="*80)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'─'*80}")
            print(f"Model: {model_name}")
            print(f"{'─'*80}")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"✓ Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"✓ Recall:    {recall:.4f} ({recall*100:.2f}%)")
            print(f"✓ F1-Score:  {f1:.4f}")
            
            # Classification Report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred, zero_division=0))
        
        # Summary Comparison
        print("\n" + "="*80)
        print("📊 SUMMARY COMPARISON")
        print("="*80)
        results_df = pd.DataFrame(results).T
        print(results_df.to_string())
        print("\n✓ Best Model by Accuracy:", results_df['accuracy'].idxmax())
        
        return results
    
    def save_best_model(self):
        """Save the best performing model"""
        # Evaluate and get best model
        y_pred_dict = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            y_pred_dict[model_name] = acc
        
        best_model_name = max(y_pred_dict, key=y_pred_dict.get)
        best_model = self.models[best_model_name]
        
        print(f"\n💾 Saving best model: {best_model_name}")
        
        # Save model
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save column names
        with open('columns.pkl', 'wb') as f:
            pickle.dump(self.column_names, f)
        
        # Save label encoder
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✓ Model saved: best_model.pkl")
        print(f"✓ Columns saved: columns.pkl")
        print(f"✓ Label encoder saved: label_encoder.pkl")
        
        return best_model_name


def main():
    print("\n" + "="*80)
    print("🩺 ADVANCED DISEASE DIAGNOSIS SYSTEM")
    print("="*80)
    
    # Initialize predictor
    predictor = DiseasePredictor()
    
    # Load and preprocess data
    try:
        X, y = predictor.load_and_preprocess_data('dataset.csv')
    except FileNotFoundError:
        print("❌ Error: dataset.csv not found!")
        return
    
    # Split data
    predictor.split_data(X, y)
    
    # Train all models
    predictor.train_decision_tree()
    predictor.train_bagging_models()
    predictor.train_boosting_models()
    predictor.train_voting_ensemble()
    
    # Evaluate models
    predictor.evaluate_models()
    
    # Save best model
    best_model = predictor.save_best_model()
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
