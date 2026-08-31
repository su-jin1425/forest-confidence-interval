import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, make_classification, make_regression
from sklearn.model_selection import train_test_split
import forestci as fci
import time

def run_benchmark():
    datasets = {
        "Auto MPG": {"type": "reg"},
        "California": {"type": "reg"},
        "Diabetes": {"type": "reg"},
        "Breast Cancer": {"type": "clf"},
        "Synth Hard": {"type": "clf"},
        "Synthetic Reg": {"type": "reg"}
    }
    
    # Load datasets
    mpg_data = np.genfromtxt('examples/data/auto_mpg.csv', delimiter=',', dtype="f8")
    datasets["Auto MPG"]["X"] = mpg_data[:, :-1]
    datasets["Auto MPG"]["y"] = mpg_data[:, -1]
    
    cal_X, cal_y = fetch_california_housing(return_X_y=True)
    datasets["California"]["X"], datasets["California"]["y"] = cal_X[:2000], cal_y[:2000]
    
    datasets["Diabetes"]["X"], datasets["Diabetes"]["y"] = load_diabetes(return_X_y=True)
    datasets["Breast Cancer"]["X"], datasets["Breast Cancer"]["y"] = load_breast_cancer(return_X_y=True)
    datasets["Synth Hard"]["X"], datasets["Synth Hard"]["y"] = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
    datasets["Synthetic Reg"]["X"], datasets["Synthetic Reg"]["y"] = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    results = []
    
    for name, data in datasets.items():
        print(f"Processing {name}...")
        X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.2, random_state=42)
        
        # 2000-tree reference
        if data["type"] == "reg":
            ref_rf = RandomForestRegressor(n_estimators=2000, random_state=42, n_jobs=-1)
        else:
            ref_rf = RandomForestClassifier(n_estimators=2000, random_state=42, n_jobs=-1)
            
        ref_rf.fit(X_train, y_train)
        
        # We use uncalibrated reference for ground truth comparison
        ref_var = fci.random_forest_error(ref_rf, X_train, X_test, calibrate=False)
        ref_var = np.maximum(ref_var, 0)
        
        for n_trees in [50, 100, 200]:
            if data["type"] == "reg":
                rf = RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)
            else:
                rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
            
            rf.fit(X_train, y_train)
            
            var_uncal = fci.random_forest_error(rf, X_train, X_test, calibrate=False)
            var_cal = fci.random_forest_error(rf, X_train, X_test, calibrate=True)
            
            neg_rate = np.mean(var_uncal < 0) * 100
            
            # Clip uncalibrated for fair comparison
            var_uncal_clipped = np.maximum(var_uncal, 0)
            
            var_rmse_uncal = np.sqrt(np.mean((var_uncal_clipped - ref_var)**2))
            var_rmse_cal = np.sqrt(np.mean((var_cal - ref_var)**2))
            
            improvement = (var_rmse_uncal - var_rmse_cal) / var_rmse_uncal * 100
            
            results.append({
                "Dataset": name,
                "Trees": n_trees,
                "Neg Rate (%)": f"{neg_rate:.1f}%",
                "Var RMSE (Uncal)": f"{var_rmse_uncal:.3g}",
                "Var RMSE (Cal)": f"{var_rmse_cal:.3g}",
                "Improvement": f"{improvement:+.1f}%"
            })
            
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_string(index=False))
    
    print("\nReStructuredText Table Format:")
    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 20 15 15 15 15 20")
    print("")
    print("   * - Dataset")
    print("     - Trees")
    print("     - Neg Rate (Uncal)")
    print("     - Var RMSE (Uncal)")
    print("     - Var RMSE (Cal)")
    print("     - Rel Improvement")
    
    last_dataset = None
    for _, row in df.iterrows():
        dataset_name = row['Dataset'] if row['Dataset'] != last_dataset else ""
        last_dataset = row['Dataset']
        print(f"   * - {dataset_name}")
        print(f"     - {row['Trees']}")
        print(f"     - {row['Neg Rate (%)']}")
        print(f"     - {row['Var RMSE (Uncal)']}")
        print(f"     - {row['Var RMSE (Cal)']}")
        print(f"     - {row['Improvement']}")

if __name__ == "__main__":
    start = time.time()
    run_benchmark()
    print(f"\nCompleted in {time.time()-start:.1f} seconds")
