# Set Matplotlib backend to Agg (non-interactive, thread-safe) before any imports that use Matplotlib
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg to avoid Tkinter issues

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving the model
from lime.lime_tabular import LimeTabularExplainer  # For LIME explanations
from sklearn.base import BaseEstimator, ClassifierMixin

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Define the features to use
    shared_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 
                       'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    
    # Check if all required columns exist
    missing_features = [feat for feat in shared_features + ['diagnosis'] if feat not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required columns in the CSV file: {missing_features}")
    
    # Extract features and target
    X = data[shared_features]
    y = data['diagnosis']
    
    return X, y

# Define base models with tuned parameters
def get_base_models():
    models = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, 
                                      class_weight='balanced', random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, 
                                          subsample=0.8, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, 
                               subsample=0.8, scale_pos_weight=1, random_state=42, n_jobs=-1)),
        ('lgb', LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, 
                                random_state=42, n_jobs=-1)),
        ('cb', CatBoostClassifier(iterations=300, learning_rate=0.05, depth=7, 
                                  random_state=42, verbose=False, thread_count=-1)),
        ('lr', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, 
                                  random_state=42))
    ]
    return models

# EnhancedModelWrapper class (compatible with scikit-learn)
class EnhancedModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.named_estimators_ = {}
        self.threshold = 0.3  # Updated threshold to 0.3

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        fitted_model = self.model.fit(X_df, y)
        if hasattr(fitted_model, 'named_estimators_'):
            self.named_estimators_ = fitted_model.named_estimators_
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        # Use predict_proba with threshold 0.3 for predictions
        proba = self.predict_proba(X_df)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        return self.model.predict_proba(X_df)

    def get_params(self, deep=True):
        params = {'model': self.model, 'feature_names': self.feature_names, 'threshold': self.threshold}
        if deep:
            params.update(self.model.get_params(deep=True))
        return params

    def set_params(self, **params):
        for param, value in params.items():
            if param in ['model', 'feature_names', 'threshold']:
                setattr(self, param, value)
            else:
                self.model.set_params(**{param: value})
        return self

# LIME explanation function (with aggregate importance)
def explain_with_lime(model, X_train, X_test, y_test, feature_names, num_instances=10):
    # Convert data to NumPy arrays for LIME
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    
    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        X_train_np, 
        feature_names=feature_names, 
        class_names=['Benign', 'Malignant'], 
        mode='classification',
        discretize_continuous=True
    )
    
    # Initialize feature importance dictionary
    feature_importance = {feature: 0 for feature in feature_names}
    
    # Select instances to explain (balanced between benign and malignant)
    benign_indices = y_test[y_test == 0].index[:num_instances//2]
    malignant_indices = y_test[y_test == 1].index[:num_instances//2]
    selected_indices = list(benign_indices) + list(malignant_indices)
    
    # Generate HTML report
    html_content = """
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; }
            h3 { color: #7f8c8d; }
        </style>
    </head>
    <body>
    <h1>LIME Analysis Report</h1>
    <h2>Individual Instance Explanations</h2>
    """
    
    # Generate explanations for selected instances
    for i, idx in enumerate(selected_indices):
        # Get explanation for current instance
        exp = explainer.explain_instance(
            X_test_np[X_test.index.get_loc(idx)], 
            model.predict_proba, 
            num_features=6
        )
        
        # Save individual explanation plot
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.title(f'LIME Explanation for Instance {i+1} (Index: {idx})')
        plt.tight_layout()
        plt.savefig(f'lime_explanation_instance_{i+1}_idx_{idx}.png')
        plt.close()  # Close the figure to free memory
        
        # Add to HTML report
        html_content += f"<h3>Instance {i+1} (Index: {idx})</h3>"
        html_content += exp.as_html()
        
        # Update feature importance
        for feat, imp in exp.as_list():
            # Extract feature name (e.g., "perimeter_mean <= -0.44" -> "perimeter_mean")
            feature_name = feat.split(' ')[0]
            if feature_name in feature_importance:
                feature_importance[feature_name] += abs(imp)
    
    # Finalize HTML report
    html_content += "</body></html>"
    with open('lime_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Create aggregate feature importance plot
    plt.figure(figsize=(12, 8))
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    # Sort features by importance and keep top 6
    sorted_idx = np.argsort(importances)[-6:]
    pos = np.arange(len(sorted_idx)) + 0.5
    
    plt.barh(pos, np.array(importances)[sorted_idx])
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.xlabel('Aggregate Absolute Importance')
    plt.title('LIME Aggregate Feature Importance Across Instances')
    plt.tight_layout()
    plt.savefig('lime_aggregate_importance.png')
    plt.close()  

# Main execution
def main():
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data('aligned_breast_cancer_dataset.csv')
        feature_names = X.columns.tolist()

        # Normalize the dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

        # Define and train stacking classifier
        base_models = get_base_models()
        meta_model = LogisticRegression(random_state=42)
        stacking_clf = StackingClassifier(
            estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
        
        # Wrap the StackingClassifier with EnhancedModelWrapper
        wrapped_clf = EnhancedModelWrapper(stacking_clf, feature_names)
        wrapped_clf.fit(X_train, y_train)

        # Save the model to .pkl format
        joblib.dump(wrapped_clf, 'stacking_classifier.pkl')

        # Evaluate on test set with threshold 0.3
        y_pred = wrapped_clf.predict(X_test)  # Uses threshold 0.3 as defined in EnhancedModelWrapper
        y_pred_proba = wrapped_clf.predict_proba(X_test)  # Get probabilities for ROC

        # Compute accuracy score
        accuracy = accuracy_score(y_test, y_pred)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (6 Shared Features, Threshold 0.3)')
        plt.savefig('confusion_matrix_6_features_threshold_0.3.png')
        plt.close()

        # Cross-validation with threshold 0.3
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Custom cross-validation to apply threshold 0.3
        cv_accuracies = []
        cv_f1_macros = []
        cv_precision_macros = []
        cv_recall_macros = []
        
        for train_idx, val_idx in cv.split(X_balanced, y_balanced):
            X_train_fold = X_balanced.iloc[train_idx]
            y_train_fold = y_balanced.iloc[train_idx]
            X_val_fold = X_balanced.iloc[val_idx]
            y_val_fold = y_balanced.iloc[val_idx]

            wrapped_clf.fit(X_train_fold, y_train_fold)
            y_pred_proba_fold = wrapped_clf.predict_proba(X_val_fold)[:, 1]
            y_pred_fold = (y_pred_proba_fold >= 0.3).astype(int)
            
            cv_accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
            report = classification_report(y_val_fold, y_pred_fold, output_dict=True)
            cv_f1_macros.append(report['macro avg']['f1-score'])
            cv_precision_macros.append(report['macro avg']['precision'])
            cv_recall_macros.append(report['macro avg']['recall'])

        # Compute mean and standard deviation for cross-validation
        cv_accuracy_mean = np.mean(cv_accuracies)
        cv_accuracy_std = np.std(cv_accuracies) * 2
        cv_f1_macro_mean = np.mean(cv_f1_macros)
        cv_f1_macro_std = np.std(cv_f1_macros) * 2
        cv_precision_macro_mean = np.mean(cv_precision_macros)
        cv_precision_macro_std = np.std(cv_precision_macros) * 2
        cv_recall_macro_mean = np.mean(cv_recall_macros)
        cv_recall_macro_std = np.std(cv_recall_macros) * 2

        # ROC and Precision-Recall Curves with 10-fold CV
        n_folds = 10
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # For ROC
        tprs = []
        fprs = []
        rocs = []
        mean_fpr = np.linspace(0, 1, 100)
        fold_fprs = []
        fold_tprs = []

        # For Precision-Recall
        precisions = []
        recalls = []
        pr_aucs = []
        mean_recall = np.linspace(0, 1, 100)
        fold_recalls = []
        fold_precisions = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_balanced, y_balanced)):
            X_train_fold = X_balanced.iloc[train_idx]
            y_train_fold = y_balanced.iloc[train_idx]
            X_val_fold = X_balanced.iloc[val_idx]
            y_val_fold = y_balanced.iloc[val_idx]

            wrapped_clf.fit(X_train_fold, y_train_fold)
            y_pred_proba_fold = wrapped_clf.predict_proba(X_val_fold)[:, 1]

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_val_fold, y_pred_proba_fold)
            roc_auc = auc(fpr, tpr)
            rocs.append(roc_auc)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            fold_fprs.append(fpr)
            fold_tprs.append(tpr)

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_val_fold, y_pred_proba_fold)
            pr_auc = auc(recall, precision)
            pr_aucs.append(pr_auc)
            precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(precision_interp)
            fold_recalls.append(recall)
            fold_precisions.append(precision)

        # ROC Curve Plot
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_roc_auc = auc(mean_fpr, mean_tpr)
        std_roc_auc = np.std(rocs)

        plt.figure(figsize=(8, 6))
        for fold in range(n_folds):
            plt.plot(fold_fprs[fold], fold_tprs[fold], color='gray', alpha=0.2, lw=1, 
                     label='Individual Folds' if fold == 0 else None)
        plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, 
                 label=f'Mean ROC (AUC = {mean_roc_auc:.2f} ± {std_roc_auc:.2f})')
        plt.fill_between(mean_fpr, mean_tpr - np.std(tprs, axis=0), mean_tpr + np.std(tprs, axis=0), 
                         color='blue', alpha=0.1, label='±1 Std. Dev.')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('10-Fold Cross-Validation ROC Curve', fontsize=14, pad=15)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig('roc_curve_10fold_academic.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Precision-Recall Curve Plot
        mean_precision = np.mean(precisions, axis=0)
        mean_pr_auc = auc(mean_recall, mean_precision)
        std_pr_auc = np.std(pr_aucs)

        plt.figure(figsize=(8, 6))
        for fold in range(n_folds):
            plt.plot(fold_recalls[fold], fold_precisions[fold], color='gray', alpha=0.2, lw=1, 
                     label='Individual Folds' if fold == 0 else None)
        plt.plot(mean_recall, mean_precision, color='red', lw=2, 
                 label=f'Mean PR Curve (AUC = {mean_pr_auc:.2f} ± {std_pr_auc:.2f})')
        plt.fill_between(mean_recall, mean_precision - np.std(precisions, axis=0), mean_precision + np.std(precisions, axis=0), 
                         color='red', alpha=0.1, label='±1 Std. Dev.')
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('10-Fold Cross-Validation Precision-Recall Curve', fontsize=14, pad=15)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig('precision_recall_curve_10fold_academic.png', dpi=300, bbox_inches='tight')
        plt.close()

        # LIME Analysis
        explain_with_lime(wrapped_clf, X_train, X_test, y_test, feature_names, num_instances=10)

        # Print all results at the end
        print("=== Final Results ===")
        print(f"Accuracy Score on Test Set (Threshold 0.3): {accuracy:.3f}")
        print("\nClassification Report (Threshold 0.3):")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix (Threshold 0.3):")
        print(cm)
        print("\nCross-Validation Results (Threshold 0.3):")
        print(f"Accuracy: {cv_accuracy_mean:.3f} (±{cv_accuracy_std:.3f})")
        print(f"F1-Macro: {cv_f1_macro_mean:.3f} (±{cv_f1_macro_std:.3f})")
        print(f"Precision-Macro: {cv_precision_macro_mean:.3f} (±{cv_precision_macro_std:.3f})")
        print(f"Recall-Macro: {cv_recall_macro_mean:.3f} (±{cv_recall_macro_std:.3f})")
        print("\nROC and Precision-Recall AUC (10-Fold CV):")
        print(f"Mean ROC AUC: {mean_roc_auc:.2f} (±{std_roc_auc:.2f})")
        print(f"Mean PR AUC: {mean_pr_auc:.2f} (±{std_pr_auc:.2f})")

    finally:
        # Ensure proper cleanup of joblib resources
        joblib.parallel._backend = None  # Reset joblib backend to avoid resource leaks

if __name__ == "__main__":
    main()