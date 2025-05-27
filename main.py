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
    pA_To_nA = 0.001
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
    #import_folder = "/Users/stefanhallermann/Desktop/"
    import_folder = "/Volumes/D/tmp/Divya"
    #filename = ("25-05-20-60Hz.xlsx")
    filename = ("25-05-21-ok6bPAC-60Hz.xlsx")

    # Format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_folder = os.path.join(import_folder, f"export_{timestamp}")
    os.makedirs(export_folder, exist_ok=True)
    export_folder_traces = os.path.join(import_folder, f"export_{timestamp}/traces")
    os.makedirs(export_folder_traces, exist_ok=True)
    export_folder_used_input = os.path.join(import_folder, f"export_{timestamp}/used_input")
    os.makedirs(export_folder_used_input, exist_ok=True)

    repo_url, commit_hash = get_git_info()
    # Save to file
    os.makedirs(export_folder, exist_ok=True)

    with open(os.path.join(export_folder, "git_version_info.txt"), "w") as f:
        f.write(f"Repository: {repo_url}\n")
        f.write(f"Commit: {commit_hash}\n")
    #print(f"Saved Git info: {repo_url}, commit {commit_hash}")

    # === IMPORT DATA ===
    print("Importing traces... ", end="", flush=True)
    df = pd.read_excel(os.path.join(import_folder, filename))
    time = df.iloc[:, 0].values  # first column = time
    time *= ms_To_s   # 2nd column = stimulation trace
    stim_signal = df.iloc[:, 1].values
    traces = df.iloc[:, 2:]  # remaining columns = traces (is still a data frame, maybe faster with .values, which returns a "D numpy array, without lables)

    df.to_parquet(os.path.join(export_folder_used_input, "my_data.parquet"))
    # for later import use: df = pd.read_parquet("my_data.parquet")
    print("done")

    # Assumes the stim file has the same time base as the main data
    # Find onset times: 0 -> 1 transitions
    stim_onsets = np.where((stim_signal[:-1] < 25) & (stim_signal[1:] >= 25))[0] + 1
    time_of_stim = time[stim_onsets]

    plt.figure()
    plt.plot(time, stim_signal, label="Stimulus Signal")
    plt.scatter(time[stim_onsets], stim_signal[stim_onsets], color='red', label="Detected Onsets")
    plt.xlabel("Time (s)")
    plt.ylabel("Stim Value")
    plt.title("Stimulus Detection")
    plt.legend()
    plt.savefig(os.path.join(export_folder, f"stimDetect.pdf"))
    plt.close()
    #print("Detected stimulation times (s):", time_of_stim)


    # for collecting the results
    n_stim = len(time_of_stim)
    results_peak = {
        "stimulus number": list(range(1, n_stim + 1)),
        "stimulus time (s)": list(time_of_stim)
    }
    results_phasic = {
        "stimulus number": list(range(1, n_stim + 1)),
        "stimulus time (s)": list(time_of_stim)
    }
    results_tonic = {
        "stimulus number": list(range(1, n_stim + 1)),
        "stimulus time (s)": list(time_of_stim)
    }
    results_charge = {
        "stimulus number": list(range(1, n_stim + 1)),
        "stimulus time (s)": list(time_of_stim)
    }

    trace_count = 0
    print("Analyzing trace:", end="", flush=True)
    for trace_name in traces.columns:
        trace_count += 1
        print(f" {trace_count}", end="", flush=True)
        original_y = pA_To_nA * traces[trace_name].values
        y = original_y.copy()

        # Calculate trace baseline (across full trace base interval)
        trace_base_mask = (time >= trace_base_st) & (time <= trace_base_end)
        trace_base = np.mean(y[trace_base_mask])

        y = y - trace_base

        peak_vals = []
        base_vals = []
        charge_vals = []

        # Stimulus-by-stimulus analysis
        for stim_time in time_of_stim:
            # Remove stimulation artifact by linear interpolation
            idx1 = np.searchsorted(time, stim_time + blank_st)
            idx2 = np.searchsorted(time, stim_time + blank_end)

            i1 = y[idx1]
            i2 = y[idx2]
            interp = np.linspace(0, 1, idx2 - idx1 + 1)
            y[idx1:idx2 + 1] = i1 + interp * (i2 - i1)

            # Calculate base (mean or min in peak interval; )
            base_mask = (time >= stim_time + base_st) & (time <= stim_time + base_end)
            #base_val = (np.mean(y[base_mask]) - trace_base)
            base_val = (np.min(y[base_mask]) - trace_base)
            base_vals.append(base_val)

            # Calculate peak (min in peak interval)
            peak_mask = (time >= stim_time + peak_st) & (time <= stim_time + peak_end)
            peak_val = (np.min(y[peak_mask]) - trace_base)
            peak_vals.append(peak_val)

            # Calculate charge (trapezoidal integration over the charge interval)
            charge_mask = (time >= stim_time + charge_start) & (time <= stim_time + charge_end)
            charge_val = np.trapezoid(y[charge_mask], time[charge_mask])  # result in nA·s
            charge_vals.append(charge_val)

        # Save all plots in one vertical layout
        fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=False)

        # 1. Original trace
        axs[0].plot(time, original_y, label="Original")
        axs[0].set_title(f"Original Trace: {trace_name}")
        axs[0].set_ylabel("Value")

        # 2. Zoomed version
        zoom_mask = (time >= zoomStart1) & (time <= zoomEnd1)
        axs[1].plot(time[zoom_mask], original_y[zoom_mask], label=f"Zoomed ({zoomStart1}–{zoomEnd1}s)", color='green')
        axs[1].set_title(f"Zoomed Artifact-Removed Trace ({zoomStart1}–{zoomEnd1}s)")
        axs[1].set_ylabel("Value")

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

        # Tight layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(export_folder_traces, f"{trace_name}.pdf"))
        plt.close()

        # Collect results
        results_tonic[trace_name] = base_vals
        results_peak[trace_name] = peak_vals
        results_phasic[trace_name] = np.array(peak_vals) - np.array(base_vals)
        results_charge[trace_name] = charge_vals

    print("done")

    # Export analysis results
    tmp_df = pd.DataFrame(results_tonic)
    tmp_df.to_excel(os.path.join(export_folder, "results_tonic.xlsx"), index=False)
    tmp_df = pd.DataFrame(results_peak)
    tmp_df.to_excel(os.path.join(export_folder, "results_peak.xlsx"), index=False)
    tmp_df = pd.DataFrame(results_phasic)
    tmp_df.to_excel(os.path.join(export_folder, "results_phasic.xlsx"), index=False)
    tmp_df = pd.DataFrame(results_charge)
    tmp_df.to_excel(os.path.join(export_folder, "results_charge.xlsx"), index=False)

if __name__ == '__main__':
    CmEval()

