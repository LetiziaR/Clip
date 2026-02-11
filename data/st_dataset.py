import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.ptbxl_dataset import PTBXL
import torch


root = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra59ver2/ptb-xl-project/files/ptb-xl/1.0.3"

# folds to check for fold-based splits
folds_to_test = [1, 2, 10]  

# Sampling rates to test 
sampling_rates = [100, 500]

# -------------------------
# Function to analyze a dataset
# -------------------------
def analyze_dataset(ds, label=""):
    """
    Analyze a PTB-XL dataset and print summary statistics.

    Args:
        ds: PTBXL dataset object
        label: string to identify dataset in output
    """
    print(f"\n--- Analysis for {label} ---")
    print(f"Number of ECGs: {len(ds)}")

    lengths = []           # signal length per ECG
    nan_count = 0          # total NaN timestamps
    all_min = []           # min per ECG  (should we normalize per lead??)
    all_max = []           # max per ECG 
    sum_per_lead = torch.zeros(12)  # for mean calculation per lead
    count_per_lead = torch.zeros(12)  # number of samples per lead

    # Loop through all ECGs in dataset
    for i, (x, text) in enumerate(ds):
        lengths.append(x.shape[0])  # collect signal length
        nan_count += torch.isnan(x).any(dim=1).sum().item()  # count NaNs per timestamp
        
        # Track global min/max 
        all_min.append(x.min().item())
        all_max.append(x.max().item())
        
        # Sum per lead for mean calculation
        sum_per_lead += x.sum(dim=0)
        count_per_lead += x.shape[0]

        # Optional progress print every 500 ECGs
        if (i+1) % 500 == 0:
            print(f"Processed {i+1}/{len(ds)} ECGs...")


    print(f"Signal lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
    print(f"Total timestamps with NaNs: {nan_count}")
    print(f"Global signal min: {min(all_min):.3f}, max: {max(all_max):.3f}")

    
    mean_per_lead = sum_per_lead / count_per_lead
    print("Mean per lead:", mean_per_lead)

    # -------------------------
    # Fold check
    # -------------------------
    for fold in folds_to_test:
        # Create a dataset with a specific fold
        ds_fold = PTBXL(root=root, sampling_rate=ds.records[0].split("_")[1], folds=[fold])
        print(f"Fold {fold} length: {len(ds_fold)}")

# -------------------------
# Run analysis for each sampling rate
# -------------------------
for sr in sampling_rates:
    ds = PTBXL(root=root, sampling_rate=sr)
    
    analyze_dataset(ds, label=f"{sr} Hz")
