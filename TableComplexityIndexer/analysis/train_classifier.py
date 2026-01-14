import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, recall_score, precision_score

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.extract_gt_features import extract_features 

def main():
    print("=== Training Multi-Feature Classifier (All Features) ===")
    
    results_path = os.path.join(os.path.dirname(__file__), "../output/processing_results.json")
    if not os.path.exists(results_path):
        print("No results found.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    X = []
    y = []
    feature_names = [
        'num_rows', 'num_cols', 'num_cells', 
        'max_rowspan', 'max_colspan', 'total_merged_cells', 'merged_cell_density',
        'header_depth', 'num_header_cells',
        'num_nested_tables', 'max_nested_depth',
        'empty_cell_ratio', 'total_text_len', 'avg_text_len', 'canonical_html_len'
    ]
    
    print(f"Extracting features for {len(data)} samples...")
    for item in data:
        teds = item.get('teds_score', 0)
        is_fail = 1 if teds < 0.99 else 0
        
        pred_html = item.get('pred_html', '')
        feats = extract_features(pred_html, item.get('filename'))
        
        if feats:
            row_vec = [feats.get(k, 0) for k in feature_names]
            X.append(row_vec)
            y.append(is_fail)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset: {len(y)} samples. Failures: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    # Models to Try
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    # 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': 'f1'
    }
    
    print("\n--- Cross-Validation Performance ---")
    
    best_model_name = None
    best_f1 = 0
    
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        avg_rec = np.mean(scores['test_recall'])
        avg_prec = np.mean(scores['test_precision'])
        avg_f1 = np.mean(scores['test_f1'])
        
        print(f"\nModel: {name}")
        print(f"  Recall (Safety):    {avg_rec:.2f}")
        print(f"  Precision (Cost):   {avg_prec:.2f}")
        print(f"  F1 Score:           {avg_f1:.2f}")
         
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name}")
    
    # Final Training on Full Data to inspect weights (if LR)
    if "Logistic" in best_model_name:
        clf = models["Logistic Regression"]
        clf.fit(X, y)
        coefs = clf.named_steps['logisticregression'].coef_[0]
        
        print("\n--- Feature Importance (Logistic Regression Coefficients) ---")
        indices = np.argsort(abs(coefs))[::-1]
        for i in indices:
            print(f"{feature_names[i]:<20}: {coefs[i]:.4f}")

if __name__ == "__main__":
    main()
