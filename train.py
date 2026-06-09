import os
import sys
import mlflow
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from src.validate import validate_github_issues
from src.pipeline import create_model_pipeline

def run_training(csv_path: str):
    print(f"Loading and validating data from {csv_path}...")
    
    # 1. Load validated Pydantic objects 
    # (We exhaust the iterator into a list so it can be passed to scikit-learn)
    issues = list(validate_github_issues(csv_path))
    
    if not issues:
        print(f"No valid data found in {csv_path}. Exiting.")
        sys.exit(1)

    print(f"Successfully loaded {len(issues)} validated issues.")

    # 2. Extract features (X) and target (y)
    X = issues
    
    # Note: The GithubIssue schema currently lacks a target 'label' field. 
    # To make this baseline run functional, we derive a dummy heuristic label 
    # from the title (e.g., classifying as "bug" vs "other").
    y = ["bug" if "bug" in issue.issue_title.lower() else "other" for issue in issues]

    # 3. Split the dataset into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set MLflow experiment to group our runs together
    mlflow.set_experiment("github_issue_classifier")

    with mlflow.start_run():
        # Enable scikit-learn autologging to automatically capture hyperparameters
        mlflow.sklearn.autolog()

        # 4. Create and train the end-to-end pipeline
        print("Initializing model pipeline...")
        pipeline = create_model_pipeline()
        
        # Explicitly log key hyperparameters
        pipeline_params = pipeline.get_params()
        mlflow.log_param("vectorizer_max_features", pipeline_params.get("vectorizer__max_features"))
        mlflow.log_param("rf_n_estimators", pipeline_params.get("classifier__n_estimators"))

        print("Training the Random Forest model...")
        pipeline.fit(X_train, y_train)
        
        # 5. Evaluate the model on both train and test sets
        print("Evaluating model baseline...")
        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)
        
        # Explicitly log our custom metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Compute and log Precision, Recall, and F1 (treating 'bug' as the positive class)
        precision = precision_score(y_test, test_preds, pos_label="bug")
        recall = recall_score(y_test, test_preds, pos_label="bug")
        f1 = f1_score(y_test, test_preds, pos_label="bug")
        
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy:  {test_accuracy:.4f}")
        print(f"F1 Score:          {f1:.4f}")
        
        # Serialize the pipeline and register it as an MLflow artifact
        print("Serializing and registering pipeline artifact...")
        joblib.dump(pipeline, "pipeline.joblib")
        mlflow.log_artifact("pipeline.joblib")

if __name__ == "__main__":
    csv_file = "data/raw/smaller.csv"
    if not os.path.exists(csv_file):
        print(f"Warning: '{csv_file}' not found in the current directory.")
        
    run_training(csv_file)
