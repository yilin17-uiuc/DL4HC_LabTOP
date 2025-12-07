import argparse
import os
import shutil
import torch
import pandas as pd
from transformers import AutoTokenizer

# Import local modules
from labtop_dataset import LabTOPDataset
from labtop_model import LabTOPModel

def create_dummy_data(root_dir):
    """Creates dummy CSV files for testing."""
    os.makedirs(root_dir, exist_ok=True)
    print(f"Creating dummy data in {root_dir}...")
    
    # Create minimal necessary files for LabTOPDataset
    # 1. patients
    pd.DataFrame({
        "subject_id": [1001, 1002], "gender": ["F", "M"], "anchor_age": [50, 65]
    }).to_csv(os.path.join(root_dir, "patients.csv.gz"), index=False, compression="gzip")
    
    # 2. admissions
    pd.DataFrame({
        "subject_id": [1001, 1002], "hadm_id": [2001, 2002], "race": ["WHITE", "ASIAN"]
    }).to_csv(os.path.join(root_dir, "admissions.csv.gz"), index=False, compression="gzip")
    
    # 3. icustays
    pd.DataFrame({
        "subject_id": [1001, 1002], "hadm_id": [2001, 2002], "stay_id": [3001, 3002],
        "intime": ["2150-01-01 10:00:00", "2150-02-01 12:00:00"],
        "outtime": ["2150-01-05 10:00:00", "2150-02-05 12:00:00"]
    }).to_csv(os.path.join(root_dir, "icustays.csv.gz"), index=False, compression="gzip")
    
    # 4. labevents
    pd.DataFrame({
        "subject_id": [1001], "hadm_id": [2001], "itemid": [50800],
        "charttime": ["2150-01-01 11:00:00"], "valuenum": [7.4], "valueuom": ["units"]
    }).to_csv(os.path.join(root_dir, "labevents.csv.gz"), index=False, compression="gzip")

    # 5. inputevents
    pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "amount", "amountuom"]).to_csv(
        os.path.join(root_dir, "inputevents.csv.gz"), index=False, compression="gzip"
    )
    
    # 6. outputevents
    pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "value", "valueuom"]).to_csv(
        os.path.join(root_dir, "outputevents.csv.gz"), index=False, compression="gzip"
    )
    
    # 7. procedureevents
    pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "value", "valueuom"]).to_csv(
        os.path.join(root_dir, "procedureevents.csv.gz"), index=False, compression="gzip"
    )

def main():
    parser = argparse.ArgumentParser(description="Run LabTOP Model")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, default="MIMIC-IV", help="Dataset name (e.g., MIMIC-IV, eICU)")
    parser.add_argument("--data_root", type=str, default="dummy_data", help="Path to raw data")
    
    # Task args (Placeholder for now)
    parser.add_argument("--task", type=str, default="Pretraining", help="Task name")
    
    # Model args
    parser.add_argument("--model_type", type=str, default="labtop", help="Model type")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_model", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size (if applicable)")
    parser.add_argument("--no_temp_kernels", type=int, default=10, help="Number of temporal kernels")
    parser.add_argument("--point_size", type=int, default=10, help="Point size")
    parser.add_argument("--last_linear_size", type=int, default=20, help="Last linear layer size")
    parser.add_argument("--diagnosis_size", type=int, default=20, help="Diagnosis embedding size")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--main_dropout_rate", type=float, default=0.3, help="Main dropout rate")
    parser.add_argument("--temp_dropout_rate", type=float, default=0.1, help="Temporal dropout rate")
    
    args = parser.parse_args()
    
    print(f"\n=== Running LabTOP with the following configuration ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=======================================================\n")
    
    # Setup dummy data if using default
    if args.data_root == "dummy_data":
        if os.path.exists(args.data_root):
            shutil.rmtree(args.data_root)
        create_dummy_data(args.data_root)
        
    # 1. Process Dataset
    print("\n[1/3] Processing Dataset...")
    dataset = LabTOPDataset(
        root=args.data_root,
        save_path="processed_data",
        stay_limit=100
    )
    dataset.process()
    
    # 2. Initialize Model
    print("\n[2/3] Initializing Model...")
    tokenizer = AutoTokenizer.from_pretrained("processed_data")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = LabTOPModel(
        tokenizer=tokenizer,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.n_layers,
        dropout=args.main_dropout_rate
    )
    print(f"Model initialized: {model}")
    
    # 3. Dummy Forward Pass
    print("\n[3/3] Running Dummy Forward Pass...")
    dummy_input = torch.randint(0, len(tokenizer), (2, 128)) # Batch=2, Seq=128
    outputs = model(dummy_input)
    print(f"Output logits shape: {outputs.logits.shape}")
    
    print("\nDone! Test run completed successfully.")
    
    # Cleanup
    if args.data_root == "dummy_data":
        shutil.rmtree(args.data_root)

if __name__ == "__main__":
    main()
