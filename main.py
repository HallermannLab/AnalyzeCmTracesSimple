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
    from scipy.optimize import curve_fit

    # Constants (can be adjusted)
    F_To_pF = 1e12

    t0 = 10.0 # in seconds

    trace_base_st = -9  # in seconds
    trace_base_end = -1

    fit_st = 1
    fit_end = 15

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


    original_time = df.iloc[:, 0].values  # first column = time
    time = original_time.copy()
    time = time - t0
    traces = df.iloc[:, 1:]  # remaining columns = traces (is still a data frame, maybe faster with .values, which returns a "D numpy array, without lables)

    #df.to_parquet(os.path.join(output_folder_used_input, "my_data.parquet"))
    # for later import use: df = pd.read_parquet("my_data.parquet")

    # Prepare results table structure
    fit_results = {
        "traceName": traceName,
        "solution": solution,
        "sequence": sequence,
        "amplitude": [],
        "tau": []
    }

    trace_count = 0
    print("Analyzing trace:", end="", flush=True)
    for trace_name in traceName:
        trace_count += 1
        print(f" {trace_count}", end="", flush=True)
        original_y = F_To_pF * traces.iloc[:, trace_count - 1].values
        y = original_y.copy()

        #for checking the area used for fitting and baseline
        """
        plt.plot(time, y)
        plt.axvspan(trace_base_st, trace_base_end, color='red', alpha=0.3, label='Baseline')
        plt.axvspan(fit_st, fit_end, color='green', alpha=0.3, label='Fit')
        plt.legend()
        plt.title(trace_name)
        plt.show()
        """

        # --- 1. Baseline Subtraction ---
        # Extract baseline indices
        baseline_mask = (time >= trace_base_st) & (time <= trace_base_end)
        baseline_time = time[baseline_mask]
        baseline_values = y[baseline_mask]

        # Fit linear function: y = m*x + b
        coeffs = np.polyfit(baseline_time, baseline_values, deg=1)
        baseline_fit_line = np.polyval(coeffs, time)
        y_baseline_subtracted = y - baseline_fit_line

        # --- 2. Exponential Fit ---
        def exp_func(t, A, tau):
            return A * np.exp(-t / tau)

        # Fit exponential from fit_st to fit_end
        fit_mask = (time >= fit_st) & (time <= fit_end)
        try:
            popt, _ = curve_fit(exp_func, time[fit_mask], y_baseline_subtracted[fit_mask],
                                p0=(np.max(y_baseline_subtracted), 5))
            A_fit, tau_fit = popt
        except Exception as e:
            print(f"\nFit failed for trace {trace_name}: {e}")
            A_fit, tau_fit = np.nan, np.nan

        fit_results["amplitude"].append(A_fit)
        fit_results["tau"].append(tau_fit)

        # --- 3. Plotting ---
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Top: Original trace with baseline fit
        axs[0].plot(original_time, y, label="Original")
        axs[0].plot(original_time, baseline_fit_line, label="Baseline fit", linestyle="--")
        axs[0].set_title(f"{trace_name}: Original + Baseline")
        axs[0].legend()
        axs[0].set_ylabel("pF")

        # Bottom: Baseline-subtracted with exponential fit
        axs[1].plot(original_time, y_baseline_subtracted, label="Baseline-subtracted")
        if not np.isnan(A_fit):
            axs[1].plot(original_time, exp_func(time, *popt), label="Exp fit", linestyle="--")
        axs[1].set_title("Baseline-subtracted + Exp fit")
        axs[1].legend()
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("pF")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_traces, f"{trace_count:03d}_{trace_name}.pdf"))
        plt.close()

    print(" done!")

    # Export analysis results
    # Assemble DataFrame for export
    results_df = pd.DataFrame([
        fit_results["traceName"],
        fit_results["solution"],
        fit_results["sequence"],
        fit_results["amplitude"],
        fit_results["tau"]
    ])
    results_df.index = ["traceName", "solution", "sequence", "amplitude", "tau"]

    results_df.to_excel(os.path.join(output_folder, "fit_results.xlsx"), header=False)

if __name__ == '__main__':
    CmEval()

