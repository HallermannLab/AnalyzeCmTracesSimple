# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import subprocess
import os

def get_git_info():
    try:
        repo_url = subprocess.check_output(['git', 'remote', 'get-url', 'origin']).decode().strip()
    except Exception:
        repo_url = "unknown"

    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        commit_hash = "unknown"

    return repo_url, commit_hash


def CmEval():
    import os
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pyarrow
    import fastparquet

    # Constants (can be adjusted)
    F_To_pF = 1e12
    ms_To_s = 0.001
    blank_st = -0.3 * ms_To_s
    blank_end = 1.5 * ms_To_s
    base_st = -0.4 * ms_To_s
    base_end = 0.0 * ms_To_s
    peak_st = 0.5 * ms_To_s
    peak_end = 3.0 * ms_To_s
    charge_start = 0.0 * ms_To_s
    charge_end = 15.0  * ms_To_s

    trace_base_st = 0  # in seconds
    trace_base_end = 1

    zoomStart1 = 1.73 # in s
    zoomEnd1  = 1.75
    zoomStart2 = 1.7  # in s
    zoomEnd2 = 1.8

    # === CONFIGURATION ===
    ROOT_FOLDER = "/Users/stefanhallermann/Library/CloudStorage/Dropbox/tmp/Sophie"
    import_folder = os.path.join(ROOT_FOLDER, "in")
    filename = ("endo30ms.xlsx")

    # Format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(ROOT_FOLDER, f"output_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    output_folder_traces = os.path.join(ROOT_FOLDER, f"output_{timestamp}/traces")
    os.makedirs(output_folder_traces, exist_ok=True)
    output_folder_used_input = os.path.join(ROOT_FOLDER, f"output_{timestamp}/used_input")
    os.makedirs(output_folder_used_input, exist_ok=True)

    repo_url, commit_hash = get_git_info()
    # Save to file
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "git_version_info.txt"), "w") as f:
        f.write(f"Repository: {repo_url}\n")
        f.write(f"Commit: {commit_hash}\n")
    #print(f"Saved Git info: {repo_url}, commit {commit_hash}")

    # === IMPORT DATA ===
    print("Importing traces... ", end="", flush=True)
    df = pd.read_excel(os.path.join(import_folder, filename), header=[0, 1, 2])
    print("done!")

    # Extract the second and third header rows
    solution = list(df.columns.get_level_values(1)[1:])  # Skip 'time'
    sequence = list(df.columns.get_level_values(2)[1:])  # Skip 'time'
    traceName = list(df.columns.get_level_values(0)[1:])  # Skip 'time'
    # Optional: convert sequence to integers
    sequence = [int(x) for x in sequence]

    # Drop the second and third header levels (keep only first, i.e., 'file1', 'file2', etc.)
    df.columns = df.columns.droplevel([1, 2])
    # Display or return the results
    print("traceName:", traceName)
    print("solution:", solution)
    print("sequence:", sequence)
    # Drop rows where all data columns except 'time' are NaN
    df = df.dropna(subset=df.columns[1:], how='all')
    print(df.head())  # Cleaned DataFrame
    #input("Press Enter to continue...")


    time = df.iloc[:, 0].values  # first column = time
    time *= ms_To_s   # 2nd column = stimulation trace
    stim_signal = df.iloc[:, 1].values
    traces = df.iloc[:, 2:]  # remaining columns = traces (is still a data frame, maybe faster with .values, which returns a "D numpy array, without lables)

    #df.to_parquet(os.path.join(output_folder_used_input, "my_data.parquet"))
    # for later import use: df = pd.read_parquet("my_data.parquet")

    trace_count = 0
    print("Analyzing trace:", end="", flush=True)
    for trace_name in traceName:
        trace_count += 1
        print(f" {trace_count}", end="", flush=True)
        original_y = F_To_pF * traces[trace_name].values
        y = original_y.copy()

        # Save all plots in one vertical layout
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=False)

        # 1. Original trace
        axs[0].plot(time, original_y, label="Original")
        axs[0].set_title(f"Original Trace: {trace_name}")
        axs[0].set_ylabel("Value")

        # 2. Zoomed version
        zoom_mask = (time >= zoomStart1) & (time <= zoomEnd1)
        axs[1].plot(time[zoom_mask], original_y[zoom_mask], label=f"Zoomed ({zoomStart1}–{zoomEnd1}s)", color='green')
        axs[1].set_title(f"Zoomed Artifact-Removed Trace ({zoomStart1}–{zoomEnd1}s)")
        axs[1].set_ylabel("Value")


        """
        # 3. Artifact-removed trace
        axs[2].plot(time, y, label="Stim Artifact Removed", color='orange')
        axs[2].set_title("After Artifact Removal")
        axs[2].set_ylabel("Value")

        # 4. Zoomed version
        zoom_mask = (time >= zoomStart1) & (time <= zoomEnd1)
        axs[3].plot(time[zoom_mask], y[zoom_mask], label=f"Zoomed ({zoomStart1}–{zoomEnd1}s)", color='green')
        axs[3].set_title(f"Zoomed Artifact-Removed Trace ({zoomStart1}–{zoomEnd1}s)")
        axs[3].set_ylabel("Value")

        # 5. Zoomed version
        zoom_mask = (time >= zoomStart2) & (time <= zoomEnd2)
        axs[4].plot(time[zoom_mask], y[zoom_mask], label=f"Zoomed ({zoomStart2}–{zoomEnd2}s)", color='green')
        axs[4].set_title(f"Zoomed Artifact-Removed Trace ({zoomStart2}–{zoomEnd2}s)")
        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("Value")
        """

        # Tight layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_traces, f"{trace_count:03d}_{trace_name}.pdf"))
        plt.close()

        # Collect results

    print(" done!")

    # Export analysis results

if __name__ == '__main__':
    CmEval()

