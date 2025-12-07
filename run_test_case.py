import os
import pandas as pd
import numpy as np
from labtop_dataset import LabTOPDataset
import shutil

def create_dummy_mimic_data(root_dir):
    """Creates dummy MIMIC-IV CSV files for testing."""
    os.makedirs(root_dir, exist_ok=True)
    
    # 1. patients.csv.gz
    patients = pd.DataFrame({
        "subject_id": [1001, 1002],
        "gender": ["F", "M"],
        "anchor_age": [50, 65]
    })
    patients.to_csv(os.path.join(root_dir, "patients.csv.gz"), index=False, compression="gzip")
    
    # 2. admissions.csv.gz
    admissions = pd.DataFrame({
        "subject_id": [1001, 1002],
        "hadm_id": [2001, 2002],
        "race": ["WHITE", "ASIAN"]
    })
    admissions.to_csv(os.path.join(root_dir, "admissions.csv.gz"), index=False, compression="gzip")
    
    # 3. icustays.csv.gz
    icustays = pd.DataFrame({
        "subject_id": [1001, 1002],
        "hadm_id": [2001, 2002],
        "stay_id": [3001, 3002],
        "intime": ["2150-01-01 10:00:00", "2150-02-01 12:00:00"],
        "outtime": ["2150-01-05 10:00:00", "2150-02-05 12:00:00"]
    })
    icustays.to_csv(os.path.join(root_dir, "icustays.csv.gz"), index=False, compression="gzip")
    
    # 4. labevents.csv.gz
    labevents = pd.DataFrame({
        "subject_id": [1001, 1001],
        "hadm_id": [2001, 2001],
        "itemid": [50800, 50802],
        "charttime": ["2150-01-01 11:00:00", "2150-01-01 12:00:00"],
        "valuenum": [7.4, 0.5],
        "valueuom": ["units", "mmol/L"]
    })
    labevents.to_csv(os.path.join(root_dir, "labevents.csv.gz"), index=False, compression="gzip")
    
    # 5. inputevents.csv.gz
    inputevents = pd.DataFrame({
        "subject_id": [1001],
        "hadm_id": [2001],
        "stay_id": [3001],
        "itemid": [220001],
        "starttime": ["2150-01-01 13:00:00"],
        "amount": [100],
        "amountuom": ["ml"]
    })
    inputevents.to_csv(os.path.join(root_dir, "inputevents.csv.gz"), index=False, compression="gzip")
    
    # 6. outputevents.csv.gz
    outputevents = pd.DataFrame({
        "subject_id": [1001],
        "hadm_id": [2001],
        "stay_id": [3001],
        "itemid": [220002],
        "charttime": ["2150-01-01 14:00:00"],
        "value": [200],
        "valueuom": ["ml"]
    })
    outputevents.to_csv(os.path.join(root_dir, "outputevents.csv.gz"), index=False, compression="gzip")
    
    # 7. procedureevents.csv.gz
    procedureevents = pd.DataFrame({
        "subject_id": [1001],
        "hadm_id": [2001],
        "stay_id": [3001],
        "itemid": [220003],
        "starttime": ["2150-01-01 15:00:00"],
        "value": [1],
        "valueuom": ["min"]
    })
    procedureevents.to_csv(os.path.join(root_dir, "procedureevents.csv.gz"), index=False, compression="gzip")

    print(f"Dummy data created in {root_dir}")

def main():
    dummy_root = "dummy_mimic_data"
    processed_dir = "dummy_processed_data"
    
    # Clean up previous runs
    if os.path.exists(dummy_root):
        shutil.rmtree(dummy_root)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
        
    try:
        # 1. Create Data
        create_dummy_mimic_data(dummy_root)
        
        # 2. Run Preprocessing
        print("\nRunning LabTOPDataset processing...")
        dataset = LabTOPDataset(
            root=dummy_root,
            save_path=processed_dir,
            stay_limit=None
        )
        dataset.process()
        
        print("\nTest Case Passed! Data processed successfully.")
        
    except Exception as e:
        print(f"\nTest Case Failed: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(dummy_root):
            shutil.rmtree(dummy_root)
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)

if __name__ == "__main__":
    main()
