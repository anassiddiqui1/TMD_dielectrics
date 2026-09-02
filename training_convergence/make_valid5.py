from ase.io import read, write
import random
import os
import math

# Load all configurations
all_dataset = read('../training/alloys_dielectric.xyz', index=':')

# Total configurations
total_configs = len(all_dataset)
print(f"Total configurations: {total_configs}")

# Desired subset sizes
subset_sizes = [40, 80, 120, 160, 200, 220, 240, 280, 300, 320]

# Set a random seed for reproducibility
random.seed(42)

# Create output directory
output_dir = "valid5_xyz"
os.makedirs(output_dir, exist_ok=True)

# Generate 5-fold splits for each subset size
for size in subset_sizes:
    # Shuffle and pick a random subset of the given size
    subset_indices = random.sample(range(total_configs), size)
    subset = [all_dataset[i] for i in subset_indices]

    # Split into 5 nearly equal folds
    fold_size = math.ceil(size / 5)
    folds = [subset[i*fold_size : (i+1)*fold_size] for i in range(5)]

    for test_fold in range(5):
        test_set = folds[test_fold]
        train_set = [item for i, fold in enumerate(folds) if i != test_fold for item in fold]

        # Determine fold name: training is folds 1-5 minus test_fold+1
        train_fold_nums = ''.join(str(i+1) for i in range(5) if i != test_fold)
        test_fold_num = str(test_fold + 1)

        # Filenames
        train_file = os.path.join(output_dir, f"all_subset_{size}_{train_fold_nums}.xyz")
        test_file  = os.path.join(output_dir, f"all_subset_{size}_{test_fold_num}.xyz")

        write(train_file, train_set)
        write(test_file, test_set)

        print(f"Subset size {size}: Wrote train -> {train_file}, test -> {test_file}")

