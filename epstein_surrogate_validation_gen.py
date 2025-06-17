import os
import random
import warnings
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Setup
warnings.filterwarnings("ignore")
random.seed(0)

sys.path.append("C:/Users/met48/Desktop/TS-Clustering/mesa-examples-main/mesa-examples-main/examples/epstein_civil_violence/")
from epstein_civil_violence import model as epsteinCVModel

# Constants
INPUT_FILE = "valid_inputs_epstein.csv"
OUTPUT_FILE = "validation_set_dup_ecv.csv"
BATCH_SIZE = 20
MAX_ITERS = 250


# Function for a single simulation
def run_simulation(row):
    try:
        cit_dens, cop_dens, leg = row
        model = epsteinCVModel.EpsteinCivilViolence(
            citizen_density=cit_dens,
            cop_density=cop_dens,
            citizen_vision=5,
            cop_vision=5,
            legitimacy=leg,
            max_jail_term=30,
            active_threshold=0.1,
            arrest_prob_constant=2.3,
            movement=True,
            max_iters=MAX_ITERS
        )
        model.run_model()
        results = model.datacollector.get_model_vars_dataframe()
        return results['Active'].tolist()
    except Exception as e:
        return f"ERROR: {str(e)}"  # Keep output shape consistent


# Helper to check already processed rows
def get_completed_indices(output_file):
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        return set(existing.index)
    return set()


# Main parallel runner with batching and checkpointing
def generate_epstein_samples_parallel(inputs_df, output_file, n_processes=None):
    inputs_list = inputs_df.values.tolist()
    completed = get_completed_indices(output_file)
    remaining = [(i, row) for i, row in enumerate(inputs_list) if i not in completed]

    if not remaining:
        print("All inputs already processed.")
        return

    print(f"{len(remaining)} rows left to process.")

    if n_processes is None:
        n_processes = max(cpu_count() - 1, 1)

    # Prepare output file if not already
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["index"] + list(range(MAX_ITERS))).to_csv(output_file, index=False)

    with Pool(processes=n_processes) as pool:
        for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Processing"):
            batch = remaining[i:i + BATCH_SIZE]
            indices, batch_rows = zip(*batch)
            results = list(pool.map(run_simulation, batch_rows))

            # Save results immediately
            df_out = pd.DataFrame(results)
            df_out.insert(0, "index", indices)
            df_out.to_csv(output_file, mode='a', header=False, index=False)


if __name__ == "__main__":
    print("Loading inputs...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

    valid_inputs_epstein = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(valid_inputs_epstein)} input rows.")

    # Run simulations in parallel with checkpointing
    generate_epstein_samples_parallel(valid_inputs_epstein, OUTPUT_FILE)

    print("Done.")

