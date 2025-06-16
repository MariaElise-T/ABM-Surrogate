import os
import random
import warnings
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# Setup
warnings.filterwarnings("ignore")
random.seed(0)

# Append path to the custom Epstein model module
sys.path.append("C:/Users/met48/Desktop/TS-Clustering/mesa-examples-main/mesa-examples-main/examples/epstein_civil_violence/")
from epstein_civil_violence import model as epsteinCVModel

# Function for a single simulation
def run_simulation(row):
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
        max_iters=250
    )
    model.run_model()
    results = model.datacollector.get_model_vars_dataframe()
    return results['Active'].tolist()

# Main parallel runner
def generate_epstein_samples_parallel(inputs_df, n_processes=None):
    inputs_list = inputs_df.values.tolist()
    if n_processes is None:
        n_processes = cpu_count()

    print(f"Starting parallel processing with {n_processes} workers...")

    with Pool(processes=n_processes) as pool:
        results = list(pool.imap(run_simulation, inputs_list, chunksize=10))

    outputs_df = pd.DataFrame(results)
    return outputs_df

# Entry point
if __name__ == "__main__":
    print("Loading inputs...")
    input_file = "valid_inputs_epstein.csv"
    output_file = "validation_set_dup_ecv.csv"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    valid_inputs_epstein = pd.read_csv(input_file)

    print(f"Loaded {len(valid_inputs_epstein)} input rows.")

    # Run simulations in parallel
    df_epstein = generate_epstein_samples_parallel(valid_inputs_epstein)

    # Save results
    df_epstein.to_csv(output_file, index=False)
    print(f"Saved output to {output_file}")
