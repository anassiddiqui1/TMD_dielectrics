from ase.io import read, write
import random
import os

# Load all configurations
all_dataset = read('../training/alloys_dielectric.xyz', index=':')

# Total configurations
total_configs = len(all_dataset)
print(f"Total configurations: {total_configs}")

# Desired sizes
subset_sizes = [40,80,120,160,200,220,240,280,300]
# Set a random seed for reproducibility
random.seed(42)
# Shuffle indices
all_indices = list(range(total_configs))
random.shuffle(all_indices)

# Create output directory
output_dir = "subset_xyz"
os.makedirs(output_dir, exist_ok=True)

# Generate and write each subset
for size in subset_sizes:
    subset_indices = all_indices[:size]
    subset = [all_dataset[i] for i in subset_indices]
    output_file = os.path.join(output_dir, f"all_subset_{size}.xyz")
    write(output_file, subset)
    print(f"Wrote {size} configurations to {output_file}")

