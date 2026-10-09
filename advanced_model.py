"""
Advanced Disease Diagnosis Model
Implements Decision Tree, Bagging, Boosting, and Hybrid Ensemble Methods.

MEMORY-OPTIMIZED FOR LARGE DATASETS (189K+ samples, 377 features, 773 classes)
Key fixes:
- Disabled parallel processing (n_jobs=1) to prevent memory serialization errors
- Reduced estimators and max_depth to lower memory footprint
- Added subsampling and feature sampling for efficiency
"""

import pandas as pd
import numpy as np
import pickle
import warnings

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

warnings.filterwarnings("ignore")


# ==========================================================
# DATASET PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"


class DiseasePredictor:

    def __init__(self):

        self.models = {}
        self.column_names = None

        self.label_encoder = LabelEncoder()

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.target_col = None

    # ======================================================
    # LOAD AND PREPROCESS DATA
    # ======================================================

    def load_and_preprocess_data(self, filepath=DATASET_PATH):

        print("\nLoading dataset...")

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}\n"
                "Place dataset.csv beside advanced_model.py."
            )

        data = pd.read_csv(filepath)

        if data.empty:
            raise ValueError("The dataset is empty.")

        if len(data.columns) < 2:
            raise ValueError(
                "Dataset must contain features and a target column."
            )

        print("Dataset path:", filepath)
        print("Dataset shape:", data.shape)
        print("Columns:", len(data.columns))

        # Remove duplicate column names.
        data = data.loc[:, ~data.columns.duplicated()].copy()

        # Remove duplicate rows.
        initial_rows = len(data)

        data = data.drop_duplicates().reset_index(drop=True)

        print(
            "Duplicate rows removed:",
            initial_rows - len(data)
        )

        # Identify the target column.
        target_candidates = [
            col for col in data.columns
            if any(
                term in str(col).lower()
                for term in [
                    "disease",
                    "prognosis",
                    "target",
                    "label"
                ]
            )
        ]

        if not target_candidates:
            raise ValueError(
                "Target column not found. Use a column name such as "
                "Disease, Prognosis, Target, or Label."
            )

        self.target_col = target_candidates[0]

        print("Target column:", self.target_col)

        # Remove rows without a target.
        data = data.dropna(
            subset=[self.target_col]
        ).copy()

        if data.empty:
            raise ValueError(
                "No valid target values are available."
            )

        # Separate features and target.
        X = data.drop(
            columns=[self.target_col]
        ).copy()

        y = data[self.target_col].astype(str)

        # Handle missing values and categorical features.
        for col in X.columns:

            if pd.api.types.is_numeric_dtype(X[col]):

                X[col] = pd.to_numeric(
                    X[col],
                    errors="coerce"
                )

                median_value = X[col].median()

                if pd.isna(median_value):
                    median_value = 0

                X[col] = X[col].fillna(
                    median_value
                )

            else:

                X[col] = X[col].fillna(
                    "Unknown"
                ).astype(str)

        # Convert categorical feature columns to numeric columns.
        X = pd.get_dummies(X)

        # Handle infinite or missing values.
        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)

        self.column_names = X.columns.tolist()

        # Encode disease names into numerical labels.
        y = self.label_encoder.fit_transform(y)

        print("\nDataset Statistics")
        print("------------------")
        print("Total samples:", X.shape[0])
        print("Total features:", X.shape[1])

        print(
            "Disease classes:",
            len(self.label_encoder.classes_)
        )

        if len(self.label_encoder.classes_) < 2:
            raise ValueError(
                "At least two target classes are required."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "No usable feature columns found."
            )

        return X, y

    # ======================================================
    # SPLIT DATA
    # ======================================================

    def split_data(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42
    ):

        # Stratify when every class has at least two samples.
        class_counts = pd.Series(y).value_counts()

        stratify_value = (
            y
            if len(class_counts) > 1
            and class_counts.min() >= 2
            else None
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_value
        )

        print("\nTraining samples:", len(self.X_train))
        print("Testing samples:", len(self.X_test))

    # ======================================================
    # DECISION TREE
    # ======================================================

    def train_decision_tree(self):

        print("\nTraining Decision Tree...")

        model = DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced"
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        self.models["Decision Tree"] = model
        print("✓ Decision Tree training complete")

    # ======================================================
    # BAGGING MODELS
    # ======================================================

    def train_bagging_models(self):

        print("\nTraining Bagging Models...")

        # Random Forest
        print("Training Random Forest...")

        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
            class_weight="balanced"
        )

        rf.fit(
            self.X_train,
            self.y_train
        )

        self.models["Random Forest"] = rf
        print("✓ Random Forest training complete")

        # Bagging Classifier
        print("Training Bagging Classifier...")

        # Support both new and older scikit-learn versions.
        try:

            bagging = BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                n_estimators=30,
                random_state=42,
                n_jobs=1,
                max_samples=0.7,
                max_features=0.7,
                bootstrap=True
            )

        except TypeError:

            bagging = BaggingClassifier(
                base_estimator=DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                n_estimators=30,
                random_state=42,
                n_jobs=1,
                max_samples=0.7,
                max_features=0.7,
                bootstrap=True
            )

        bagging.fit(
            self.X_train,
            self.y_train
        )

        self.models["Bagging"] = bagging
        print("✓ Bagging Classifier training complete")

    # ======================================================
    # BOOSTING MODELS
    # ======================================================

    def train_boosting_models(self):

        print("\nTraining Boosting Models...")

        # AdaBoost
        print("Training AdaBoost...")

        try:

            ada = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                n_estimators=50,
                learning_rate=0.5,
                random_state=42
            )

        except TypeError:

            ada = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                n_estimators=50,
                learning_rate=0.5,
                random_state=42
            )

        ada.fit(
            self.X_train,
            self.y_train
        )

        self.models["AdaBoost"] = ada
        print("✓ AdaBoost training complete")

        # Gradient Boosting
        print("Training Gradient Boosting...")

        gb = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.7,
            max_features=0.7
        )

        gb.fit(
            self.X_train,
            self.y_train
        )

        self.models["Gradient Boosting"] = gb
        print("✓ Gradient Boosting training complete")

    # ======================================================
    # HYBRID VOTING ENSEMBLE
    # ======================================================

    def train_voting_ensemble(self):

        print("\nTraining Voting Ensemble...")

        voting_model = VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=30,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1
                    )
                ),
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=30,
                        max_depth=3,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        subsample=0.7
                    )
                ),
                (
                    "ada",
                    AdaBoostClassifier(
                        n_estimators=30,
                        learning_rate=0.5,
                        random_state=42
                    )
                )
            ],
            voting="soft"
        )

        voting_model.fit(
            self.X_train,
            self.y_train
        )

        self.models["Voting Ensemble"] = voting_model
        print("✓ Voting Ensemble training complete")

    # ======================================================
    # MODEL EVALUATION
    # ======================================================

    def evaluate_models(self):

        print("\n")
        print("=" * 75)
        print("MODEL PERFORMANCE EVALUATION")
        print("=" * 75)

        results = {}

        target_names = [
            str(name)
            for name in self.label_encoder.classes_
        ]

        labels = np.arange(
            len(target_names)
        )

        for model_name, model in self.models.items():

            print("\n" + "-" * 65)
            print("Model:", model_name)
            print("-" * 65)

            # Predictions
            y_pred = model.predict(
                self.X_test
            )

            # Performance metrics
            accuracy = accuracy_score(
                self.y_test,
                y_pred
            )

            precision = precision_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            print(
                f"Accuracy:  {accuracy:.4f} "
                f"({accuracy * 100:.2f}%)"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall:    {recall:.4f}"
            )

            print(
                f"F1-Score:  {f1:.4f}"
            )

            print("\nClassification Report (sample):")
            # Only show first 10 classes to avoid clutter
            print(
                classification_report(
                    self.y_test,
                    y_pred,
                    labels=labels[:10],
                    target_names=target_names[:10],
                    zero_division=0
                )
            )

        # Summary table
        results_df = pd.DataFrame(
            results
        ).T

        results_df = results_df.sort_values(
            by="accuracy",
            ascending=False
        )

        print("\n")
        print("=" * 75)
        print("SUMMARY COMPARISON")
        print("=" * 75)

        print(
            results_df.to_string(
                float_format=lambda value: f"{value:.4f}"
            )
        )

        best_model_name = results_df.index[0]

        print(
            "\nBest Model by Accuracy:",
            best_model_name
        )

        return results_df

    # ======================================================
    # SAVE BEST MODEL
    # ======================================================

    def save_best_model(self, results_df):

        best_model_name = (
            results_df["accuracy"].idxmax()
        )

        best_model = self.models[
            best_model_name
        ]

        print("\nSaving best model...")

        model_path = BASE_DIR / "best_model.pkl"
        columns_path = BASE_DIR / "columns.pkl"
        encoder_path = BASE_DIR / "label_encoder.pkl"
        metadata_path = BASE_DIR / "model_metadata.pkl"

        # Save trained model.
        with open(model_path, "wb") as file:

            pickle.dump(
                best_model,
                file
            )

        # Save feature column names.
        with open(columns_path, "wb") as file:

            pickle.dump(
                self.column_names,
                file
            )

        # Save disease label encoder.
        with open(encoder_path, "wb") as file:

            pickle.dump(
                self.label_encoder,
                file
            )

        # Save metadata.
        with open(metadata_path, "wb") as file:

            pickle.dump(
                {
                    "best_model_name": best_model_name,
                    "target_column": self.target_col
                },
                file
            )

        print("Best model:", best_model_name)

        print("Saved:", model_path)
        print("Saved:", columns_path)
        print("Saved:", encoder_path)
        print("Saved:", metadata_path)

        return best_model_name


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    print("\n")
    print("=" * 75)
    print("ADVANCED DISEASE DIAGNOSIS SYSTEM")
    print("=" * 75)
    print("Memory-optimized for large datasets")
    print("=" * 75)

    predictor = DiseasePredictor()

    try:

        # Load dataset.
        X, y = predictor.load_and_preprocess_data(
            DATASET_PATH
        )

        # Split dataset.
        predictor.split_data(
            X,
            y
        )

        # Train models.
        predictor.train_decision_tree()

        predictor.train_bagging_models()

        predictor.train_boosting_models()

        predictor.train_voting_ensemble()

        # Evaluate models.
        results_df = predictor.evaluate_models()

        # Save best model.
        predictor.save_best_model(
            results_df
        )

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
        MemoryError
    ) as error:

        print("\nERROR:", error)

        return

    print("\n")
    print("=" * 75)
    print("TRAINING COMPLETE!")
    print("=" * 75)

    print(
        "Note: This is an educational model, "
        "not a clinically validated diagnostic system."
    )


if __name__ == "__main__":

    main()
