# LabTOP: Label-Aware Time-Series Pre-training

This example focuses only on generating LabTOP-style token sequences and does not modify existing PyHealth datasets.It demonstrates how to preprocess **MIMIC-IV ICU data** into **LabTOP-style token sequences** using `Bio_ClinicalBERT` along with custom time and event-type tokens. It provides a clean, reproducible pipeline for transforming raw MIMIC-IV events into tokenized sequences suitable for clinical language models.

## Overview

LabTOP utilizes a GPT-2 based architecture to model medical time-series data. It treats medical events (labs, medications, procedures) as a sequence of tokens and performs autoregressive pre-training. This approach allows the model to learn rich representations of patient states and predict future values or events.


## Data Requirements

To use LabTOP, you will need access to one of the following datasets via [PhysioNet](https://physionet.org/).

### 1. MIMIC-IV (v2.0 or later)
The following tables are required in CSV format (compressed `.csv.gz`):
-   `hosp/patients.csv.gz`
-   `hosp/admissions.csv.gz`
-   `hosp/d_labitems.csv.gz`
-   `hosp/labevents.csv.gz`
-   `hosp/emar.csv.gz`
-   `hosp/emar_detail.csv.gz`
-   `icu/icustays.csv.gz`
-   `icu/d_items.csv.gz`
-   `icu/inputevents.csv.gz`
-   `icu/outputevents.csv.gz`
-   `icu/procedureevents.csv.gz`

### 2. eICU-CRD (v2.0)
The following tables are required:
-   `patient.csv.gz`
-   `lab.csv.gz`
-   `microLab.csv.gz`
-   `intakeOutput.csv.gz`
-   `infusionDrug.csv.gz`
-   `medication.csv.gz`
-   `treatment.csv.gz`

### 3. HiRID (v1.1.1)
The data should be organized as follows:
-   **Raw Stage**:
    -   `observation_tables/` (parquet or csv parts)
    -   `pharma_records/` (parquet or csv parts)
-   **Reference Data**:
    -   `hirid_variable_reference.csv`
    -   `general_table.csv`

## Installation

Ensure you have the necessary dependencies installed:

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## Runnable Test Case

We provide a script `run_labtop.py` that allows you to run the model with various configurations. It automatically generates dummy data if no data path is provided.

**Run with default settings:**
```bash
python run_labtop.py
```

**Run with custom hyperparameters (Example):**
```bash
python run_labtop.py \
    --dataset eICU \
    --task LoS \
    --model_type labtop \
    --n_layers 4 \
    --d_model 256 \
    --n_heads 4 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --main_dropout_rate 0.3
```

**Expected Output:**
```text
=== Running LabTOP with the following configuration ===
dataset: eICU
task: LoS
model_type: labtop
n_layers: 4
...
=======================================================

Creating dummy data in dummy_data...
[1/3] Processing Dataset...
...
[2/3] Initializing Model...
...
[3/3] Running Dummy Forward Pass...
Output logits shape: torch.Size([2, 128, 29043])

Done! Test run completed successfully.
```

## Contribution to PyHealth

This implementation is structured to be contributed to the [PyHealth](https://github.com/sunlabuiuc/PyHealth) library.

### Files
-   `labtop_dataset.py`: Corresponds to `pyhealth.datasets`
-   `labtop_model.py`: Corresponds to `pyhealth.models`

### Steps to Contribute
1.  Fork and clone the PyHealth repository.
2.  Copy the files to the respective directories in PyHealth.
3.  Fill in the **Name**, **NetId**, and **Paper Link** in the file headers.
4.  Submit a Pull Request to the `develop` branch.

## Reference

If you use this code, please cite the original paper:
*LabTOP: A Unified Model for Lab Test Outcome Prediction on Electronic Health Records*
