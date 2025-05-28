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

def test():
    import os
    import analyze_two_groups as myAna
    from datetime import datetime
    ROOT_FOLDER = "/Users/stefanhallermann/Library/CloudStorage/Dropbox/tmp/test"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(ROOT_FOLDER, f"output_SH_{timestamp}")
    group_a = [1.1, 2.3, 2.5, 1.8, 2.0,4,3,4,5,5,5,5,3,6]
    group_b = [3.0, 3.2, 2.9, 3.1, 2.8,6,7,8,8,7,7,7,8,8,9,9,7.4,5,5,5,5,6]
    myAna.analyze_two_groups(group_a, group_b, output_folder, group_names=["Control", "Treatment"], title="Effect Size")


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
    fit_end = 19

    # === CONFIGURATION ===
    ROOT_FOLDER = "/Users/stefanhallermann/Library/CloudStorage/Dropbox/tmp/Sophie/30ms"
    import_folder = os.path.join(ROOT_FOLDER, "in")
    filename = ("endo30ms.xlsx")

    # Format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(ROOT_FOLDER, f"output_SH_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    output_folder_traces_1exp = os.path.join(output_folder, "1exp/traces")
    os.makedirs(output_folder_traces_1exp, exist_ok=True)
    output_folder_traces_1expY = os.path.join(output_folder, "1expY/traces")
    os.makedirs(output_folder_traces_1expY, exist_ok=True)
    output_folder_traces_2exp = os.path.join(output_folder, "2exp/traces")
    os.makedirs(output_folder_traces_2exp, exist_ok=True)

    output_folder_used_input = os.path.join(output_folder, "used_input")
    os.makedirs(output_folder_used_input, exist_ok=True)

    output_folder_fitresults_1exp = os.path.join(output_folder, "1exp/fitresults")
    os.makedirs(output_folder_fitresults_1exp, exist_ok=True)
    output_folder_fitresults_1expY = os.path.join(output_folder, "1expY/fitresults")
    os.makedirs(output_folder_fitresults_1expY, exist_ok=True)
    output_folder_fitresults_2exp = os.path.join(output_folder, "2exp/fitresults")
    os.makedirs(output_folder_fitresults_2exp, exist_ok=True)

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

    df.to_excel(os.path.join(output_folder_used_input, "my_data.xlsx"))
    #if you use very large traces better use this
    #df.to_parquet(os.path.join(output_folder_used_input, "my_data.parquet"))
    # for later import use: df = pd.read_parquet("my_data.parquet")

    # Prepare results table structure
    fit_results_1exp = []
    fit_results_1expY = []
    fit_results_2exp = []

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

        # ----------------------------------
        # --- Exponential Fit without offset---
        def exp_func(t, A, tau):
            return A * np.exp(-t / tau)

        # Fit exponential from fit_st to fit_end
        fit_mask = (time >= fit_st) & (time <= fit_end)
        try:
            popt, _ = curve_fit(exp_func, time[fit_mask], y_baseline_subtracted[fit_mask],
                                p0=(np.max(y_baseline_subtracted), 5), bounds=([0, 0 ], [np.inf, np.inf]))
            A_fit, tau_fit = popt
        except Exception as e:
            print(f"\nFit failed for trace {trace_name}: {e}")
            A_fit, tau_fit = np.nan, np.nan
            popt = [np.nan, np.nan]

        fit_results_1exp.append({
            'traceName': trace_name,
            'solution': solution[trace_count - 1],  # because trace_count starts from 1
            'sequence': sequence[trace_count - 1],
            'amplitude': popt[0],
            'tau': popt[1]
        })

        # --- Plotting ---
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Top: Original trace with baseline fit
        axs[0].plot(time, y, label="Original")
        axs[0].plot(time, baseline_fit_line, label="Baseline fit", linestyle="--")
        axs[0].set_title(f"{trace_name}: Original + Baseline")
        axs[0].legend()
        axs[0].set_ylabel("pF")

        # Bottom: Baseline-subtracted with exponential fit
        fit_plot_x = time[time >= 0]
        fit_plot_y = exp_func(fit_plot_x, popt[0] , popt[1])
        axs[1].plot(time, y_baseline_subtracted, label="Baseline-subtracted")
        if not np.isnan(A_fit):
            axs[1].plot(fit_plot_x, fit_plot_y, 'r--', label="Exponential fit")
            #axs[1].plot(original_time, exp_func(time, *popt), label="Exp fit", linestyle="--")
        axs[1].set_title("Baseline-subtracted + Exp fit")
        axs[1].legend()
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("pF")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_traces_1exp, f"{trace_count:03d}_{trace_name}.pdf"))
        plt.close()

        # ----------------------------------
        # --- Exponential Fit with offset---
        def exp_funcY(t, A, tau, y0):
            return A * np.exp(-t / tau) + y0

        fit_mask = (time >= fit_st) & (time <= fit_end)
        try:
            popt, _ = curve_fit(exp_funcY, time[fit_mask], y_baseline_subtracted[fit_mask],
                                p0=(np.max(y_baseline_subtracted), 5, 0.0),bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            A_fit, tau_fit, y0_fit = popt
        except Exception as e:
            print(f"\nFit failed for trace {trace_name}: {e}")
            A_fit, tau_fit, y0_fit = np.nan, np.nan, np.nan
            popt = [np.nan, np.nan, np.nan]

        fit_results_1expY.append({
            'traceName': trace_name,
            'solution': solution[trace_count - 1],  # because trace_count starts from 1
            'sequence': sequence[trace_count - 1],
            'amplitude': popt[0],
            'tau': popt[1],
            'y0': popt[2]
        })

        # --- Plotting ---
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Top: Original trace with baseline fit
        axs[0].plot(time, y, label="Original")
        axs[0].plot(time, baseline_fit_line, label="Baseline fit", linestyle="--")
        axs[0].set_title(f"{trace_name}: Original + Baseline")
        axs[0].legend()
        axs[0].set_ylabel("pF")

        # Bottom: Baseline-subtracted with exponential fit
        fit_plot_x = time[time >= 0]
        fit_plot_y = exp_funcY(fit_plot_x, popt[0], popt[1], popt[2])
        axs[1].plot(time, y_baseline_subtracted, label="Baseline-subtracted")
        if not np.isnan(A_fit):
            axs[1].plot(fit_plot_x, fit_plot_y, 'r--', label="Exponential fit")
            #axs[1].plot(original_time, exp_func(time, *popt), label="Exp fit", linestyle="--")
        axs[1].set_title("Baseline-subtracted + Exp fit")
        axs[1].legend()
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("pF")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_traces_1expY, f"{trace_count:03d}_{trace_name}.pdf"))
        plt.close()

        # ----------------------------------
        # --- Exponential Fit with 2exp ---
        def exp_func2(t, A, tau1, aRel, tau2):
            return A * (1 - aRel) * np.exp(-t / tau1) + A * aRel * np.exp(-t / tau2)

        fit_mask = (time >= fit_st) & (time <= fit_end)
        try:
            popt, _ = curve_fit(exp_func2, time[fit_mask], y_baseline_subtracted[fit_mask],
                                p0=(np.max(y_baseline_subtracted), 2, 0.5, 10),bounds=([0, 0, 0, 0], [np.inf, np.inf, 1, np.inf]))
            A_fit, tau1_fit, aRel_fit, tau2_fit = popt
        except Exception as e:
            print(f"\nFit failed for trace {trace_name}: {e}")
            A_fit, tau1_fit, aRel_fit, tau2_fit = np.nan, np.nan, np.nan, np.nan
            popt = [np.nan, np.nan, np.nan, np.nan]

        fit_results_2exp.append({
            'traceName': trace_name,
            'solution': solution[trace_count - 1],  # because trace_count starts from 1
            'sequence': sequence[trace_count - 1],
            'amplitude': popt[0],
            'tau1': popt[1],
            'aRel': popt[2],
            'tau2': popt[3]
        })

        # --- Plotting ---
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Top: Original trace with baseline fit
        axs[0].plot(time, y, label="Original")
        axs[0].plot(time, baseline_fit_line, label="Baseline fit", linestyle="--")
        axs[0].set_title(f"{trace_name}: Original + Baseline")
        axs[0].legend()
        axs[0].set_ylabel("pF")

        # Bottom: Baseline-subtracted with exponential fit
        fit_plot_x = time[time >= 0]
        fit_plot_y = exp_func2(fit_plot_x, popt[0], popt[1], popt[2], popt[3])
        axs[1].plot(time, y_baseline_subtracted, label="Baseline-subtracted")
        if not np.isnan(A_fit):
            axs[1].plot(fit_plot_x, fit_plot_y, 'r--', label="Exponential fit")
            # axs[1].plot(original_time, exp_func(time, *popt), label="Exp fit", linestyle="--")
        axs[1].set_title("Baseline-subtracted + Exp fit")
        axs[1].legend()
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("pF")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_traces_2exp, f"{trace_count:03d}_{trace_name}.pdf"))
        plt.close()

    print(" done!")

    # Export analysis results
    # Convert collected results into a proper DataFrame
    results_df = pd.DataFrame(fit_results_1exp)

    # Save full results
    results_df.to_excel(os.path.join(output_folder_fitresults_1exp, "fit_results_all.xlsx"), index=False)

    # Save solution-separated results
    for sol in results_df['solution'].unique():
        df_sol = results_df[results_df['solution'] == sol]
        df_sol.to_excel(os.path.join(output_folder_fitresults_1exp, f"fit_results_{sol}.xlsx"), index=False)

        # Further split by sequence
        for seq in df_sol['sequence'].unique():
            df_combo = df_sol[df_sol['sequence'] == seq]
            fname = f"fit_results_{sol}_seq{seq}.xlsx"
            df_combo.to_excel(os.path.join(output_folder_fitresults_1exp, fname), index=False)

    # Export analysis results
    # Convert collected results into a proper DataFrame
    results_df = pd.DataFrame(fit_results_1expY)

    # Save full results
    results_df.to_excel(os.path.join(output_folder_fitresults_1expY, "fit_results_all.xlsx"), index=False)

    # Save solution-separated results
    for sol in results_df['solution'].unique():
        df_sol = results_df[results_df['solution'] == sol]
        df_sol.to_excel(os.path.join(output_folder_fitresults_1expY, f"fit_results_{sol}.xlsx"), index=False)

        # Further split by sequence
        for seq in df_sol['sequence'].unique():
            df_combo = df_sol[df_sol['sequence'] == seq]
            fname = f"fit_results_{sol}_seq{seq}.xlsx"
            df_combo.to_excel(os.path.join(output_folder_fitresults_1expY, fname), index=False)

    # Export analysis results
    # Convert collected results into a proper DataFrame
    results_df = pd.DataFrame(fit_results_2exp)

    # Save full results
    results_df.to_excel(os.path.join(output_folder_fitresults_2exp, "fit_results_all.xlsx"), index=False)

    # Save solution-separated results
    for sol in results_df['solution'].unique():
        df_sol = results_df[results_df['solution'] == sol]
        df_sol.to_excel(os.path.join(output_folder_fitresults_2exp, f"fit_results_{sol}.xlsx"), index=False)

        # Further split by sequence
        for seq in df_sol['sequence'].unique():
            df_combo = df_sol[df_sol['sequence'] == seq]
            fname = f"fit_results_{sol}_seq{seq}.xlsx"
            df_combo.to_excel(os.path.join(output_folder_fitresults_2exp, fname), index=False)


if __name__ == '__main__':
    CmEval()
    #test()

