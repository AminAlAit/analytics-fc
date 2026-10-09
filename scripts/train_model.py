import duckdb
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

def get_db_connection():
    base_dir = Path(__file__).resolve().parent.parent
    db_path = str(base_dir / "dbt_project" / "analytics_fc.duckdb")
    return duckdb.connect(db_path)

def load_combined_data():
    con = get_db_connection()
    # Pull both raw stats (overall, potential) AND SCDI scores for the exact same rows
    query = """
    SELECT 
        f.overall,
        f.max_potential,
        p.scdi_overall,
        p.scdi_potential,
        CASE WHEN t.short_name IS NOT NULL THEN 1 ELSE 0 END AS achieved_silverware_event
    FROM mart_pattern_maker p
    INNER JOIN stg_fifatable f ON p.player_b_id = f.player_id AND f.age = 22
    LEFT JOIN (SELECT DISTINCT short_name FROM stg_tm_trophies) t 
      ON p.player_b_name = t.short_name
    """
    df = con.execute(query).df()
    con.close()
    return df.dropna()

def train_and_log(run_name, df, features, target):
    X = df[features]
    y = df[target]
    
    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with mlflow.start_run(run_name=run_name):
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        mlflow.log_param("features", features)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc)
        
        mlflow.sklearn.log_model(clf, f"{run_name}_model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        
        print(f"[{run_name}] ROC-AUC: {roc:.4f} (Accuracy: {acc:.4f})")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    mlflow_db = f"sqlite:///{base_dir}/mlflow.db"
    mlflow.set_tracking_uri(mlflow_db)
    mlflow.set_experiment("Ultimate_Comparison_Test")
    
    print("Loading Master Dataset (5.2 million rows)...")
    df = load_combined_data()
    
    print("\\n--- STARTING THE 3-WAY COMPARISON ---")
    
    print("\\n1. Training Baseline Model (Raw Stats Only)...")
    train_and_log("1_Baseline_Model", df, ['overall', 'max_potential'], 'achieved_silverware_event')
    
    print("\\n2. Training Cascade Model (SCDI Shape Only)...")
    train_and_log("2_Cascade_Model", df, ['scdi_overall', 'scdi_potential'], 'achieved_silverware_event')
    
    print("\\n3. Training Combined Model (Stats + Shape)...")
    train_and_log("3_Combined_Model", df, ['overall', 'max_potential', 'scdi_overall', 'scdi_potential'], 'achieved_silverware_event')
