import os
import zipfile

import pandas as pd

zip_path = "/home/xyz/workspace/research-spice-shadow-price-pred/data/eia8602023.zip"
extract_path = "/home/xyz/workspace/research-spice-shadow-price-pred/data/eia8602023"

# Unzip
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Extracted to {extract_path}")

# Find the Generator file
gen_file = None
for root, _dirs, files in os.walk(extract_path):
    for file in files:
        if "3_1_Generator" in file and file.endswith(".xlsx"):
            gen_file = os.path.join(root, file)
            break

if not gen_file:
    print("Could not find Generator file (3_1_Generator*.xlsx)")
    exit(1)

print(f"Found generator file: {gen_file}")


# Find Plant file
plant_file = None
for root, _dirs, files in os.walk(extract_path):
    for file in files:
        if "2___Plant" in file and file.endswith(".xlsx"):
            plant_file = os.path.join(root, file)
            break

if not plant_file:
    print("Could not find Plant file (2___Plant*.xlsx)")
    exit(1)

print(f"Found plant file: {plant_file}")

# Load Plant File
print("Loading plant file...")
df_plant_raw = pd.read_excel(plant_file, header=None, nrows=10)

# Find header for plant
plant_header_row = None
for i, row in df_plant_raw.iterrows():
    row_values = [str(x) for x in row.values]
    if "Balancing Authority Code" in row_values and "Plant Code" in row_values:
        plant_header_row = i
        break

if plant_header_row is None:
    print("Could not find header in Plant file")
    exit(1)

df_plant = pd.read_excel(plant_file, header=plant_header_row)
df_plant.columns = df_plant.columns.astype(str).str.strip()

# Create mapping
# Ensure Plant Code is unique or handle duplicates (usually unique per plant)
# Keep only relevant columns
ba_col = [c for c in df_plant.columns if "Balancing Authority Code" in c][0]
plant_code_col = [c for c in df_plant.columns if "Plant Code" in c][0]

print(f"Mapping {plant_code_col} to {ba_col}")
# Create integer to BA string mapping
plant_to_ba = df_plant.set_index(plant_code_col)[ba_col].to_dict()

# Load Generator File
print("Loading generator file...")
# We know header is likely row 1 (0-indexed 1) based on previous run
df_gen = pd.read_excel(gen_file, header=1)
df_gen.columns = df_gen.columns.astype(str).str.strip()

# Add BA Code
print("Merging BA Codes...")
df_gen["Balancing Authority Code"] = df_gen["Plant Code"].map(plant_to_ba)

# Filter for MISO
miso_df = df_gen[df_gen["Balancing Authority Code"] == "MISO"].copy()

print(f"Found {len(miso_df)} MISO generators.")

cols_to_keep = [
    "Utility ID",
    "Utility Name",
    "Plant Code",
    "Plant Name",
    "State",
    "Generator ID",
    "Technology",
    "Prime Mover Code",
    "Energy Source 1",
    "Nameplate Capacity (MW)",
    "Operating Year",
    "Status",
    "Balancing Authority Code",
]

existing_cols = [c for c in cols_to_keep if c in miso_df.columns]
miso_df = miso_df[existing_cols]

output_csv = "/home/xyz/workspace/research-spice-shadow-price-pred/data/miso_generators_eia860_2023.csv"
miso_df.to_csv(output_csv, index=False)
print(f"Saved MISO generator data to {output_csv}")
