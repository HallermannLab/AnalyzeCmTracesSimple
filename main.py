import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow
#    import fastparquet
from scipy.optimize import curve_fit
from scipy.stats import sem  # for standard error of the mean
from scipy.signal import medfilt
from tkinter import filedialog
from tkinter import Tk
import analyze_two_groups as myAna
import git_save as myGit


def plot_group_traces(time, traces_df, traceName, solution, sequence, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Wrap metadata in a structured list for easy access
    trace_meta = [
        {"name": traceName[i], "solution": solution[i], "sequence": sequence[i], "trace": traces_df.iloc[:, i].values}
        for i in range(len(traceName))
    ]

    # Define groupings
    groupings = [
        ("all", lambda meta: True),
        ("c", lambda meta: meta["solution"] == "c"),
        ("g", lambda meta: meta["solution"] == "g"),
        ("c_seq1", lambda meta: meta["solution"] == "c" and meta["sequence"] == 1),
        ("c_seq2", lambda meta: meta["solution"] == "c" and meta["sequence"] == 2),
        ("g_seq1", lambda meta: meta["solution"] == "g" and meta["sequence"] == 1),
        ("g_seq2", lambda meta: meta["solution"] == "g" and meta["sequence"] == 2),
    ]

    for group_name, condition in groupings:
        selected = [meta for meta in trace_meta if condition(meta)]
        if not selected:
            print(f"Skipping {group_name}: no data.")
            continue

        Y = np.array([meta["trace"] for meta in selected])
        mean_trace = np.nanmean(Y, axis=0)
        sem_trace = sem(Y, axis=0, nan_policy='omit')

        # Plot
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Top: all traces
        for y in Y:
            axs[0].plot(time, y, alpha=0.5)
        axs[0].set_title(f"Superimposed Traces - {group_name}")
        axs[0].set_ylabel("Signal (pF)")
        axs[0].grid(True)

        # Bottom: mean ± SEM
        axs[1].plot(time, mean_trace, label="Mean", color="black")
        axs[1].fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace,
                            color="gray", alpha=0.4, label="SEM")
        axs[1].set_title(f"Mean ± SEM - {group_name}")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Signal (pF)")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        output_path = os.path.join(output_folder, f"traces_{group_name}.pdf")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot for group: {group_name} -> {output_path}")

def CmEval():

    root = Tk()
    root.withdraw()  # Hide the GUI window
    ROOT_FOLDER = filedialog.askdirectory(title="Select root folder which contains the 'in' Folder")
    import_folder = os.path.join(ROOT_FOLDER, "in")

    # imported parameters - must be in the "in" folder
    param_file = 'parameters.xlsx'
    param_values = pd.read_excel(os.path.join(import_folder, param_file), header=None).iloc[:,
                   1].tolist()  # second row (index 1)

    # === Assign values in order ===
    (
        filename,
        output_initials,
        F_To_pF,
        t0,
        trace_base_st,
        trace_base_end,
        fit_st,
        fit_end
    ) = param_values

    # Example: print some of the loaded parameters to confirm
    print("Import folder:", import_folder)
    print("Filename:", filename)


    # Format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d___%H-%M-%S")
    output_folder = os.path.join(ROOT_FOLDER, f"output_SH_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    output_folder_used_data_and_code = os.path.join(output_folder, "used_data_and_code")
    os.makedirs(output_folder_used_data_and_code, exist_ok=True)

    output_folder_traces_1exp = os.path.join(output_folder, "1exp/traces")
    os.makedirs(output_folder_traces_1exp, exist_ok=True)
    output_folder_traces_1expY = os.path.join(output_folder, "1expY/traces")
    os.makedirs(output_folder_traces_1expY, exist_ok=True)
    output_folder_traces_2exp = os.path.join(output_folder, "2exp/traces")
    os.makedirs(output_folder_traces_2exp, exist_ok=True)

    output_folder_fitresults_1exp = os.path.join(output_folder, "1exp/fitresults")
    os.makedirs(output_folder_fitresults_1exp, exist_ok=True)
    output_folder_fitresults_1expY = os.path.join(output_folder, "1expY/fitresults")
    os.makedirs(output_folder_fitresults_1expY, exist_ok=True)
    output_folder_fitresults_2exp = os.path.join(output_folder, "2exp/fitresults")
    os.makedirs(output_folder_fitresults_2exp, exist_ok=True)

    output_folder_parameterCompare_1exp = os.path.join(output_folder, "1exp/parameterCompare")
    os.makedirs(output_folder_parameterCompare_1exp, exist_ok=True)
    output_folder_parameterCompare_1expY = os.path.join(output_folder, "1expY/parameterCompare")
    os.makedirs(output_folder_parameterCompare_1expY, exist_ok=True)
    output_folder_parameterCompare_2exp = os.path.join(output_folder, "2exp/parameterCompare")
    os.makedirs(output_folder_parameterCompare_2exp, exist_ok=True)

    # === GIT SAVE ===
    # Provide the current script path (only works in .py, not notebooks)
    script_path = __file__ if '__file__' in globals() else None
    myGit.save_git_info(output_folder_used_data_and_code, script_path)


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

    # export used data
    df.to_excel(os.path.join(output_folder_used_data_and_code, "my_used_data.xlsx"))
    # if you use very large traces better use this
    # df.to_parquet(os.path.join(output_folder_used_data_and_code, "my_data.parquet"))
    # for later import use: df = pd.read_parquet("my_data.parquet")

    # save used parameters
    param_names = [
        "filename",
        "output_initials",
        "F_To_pF",
        "t0",
        "trace_base_st",
        "trace_base_end",
        "fit_st",
        "fit_end"
    ]
    header = ["import_folder"] + param_names
    output_values = [import_folder] + param_values
    df_export = pd.DataFrame([output_values], columns=header)
    df_export = df_export.T.reset_index()
    df_export.columns = ["parameter", "value"]
    df_export.to_excel(os.path.join(output_folder_used_data_and_code, "my_used_parameters.xlsx"), index=False)


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

        # apply median filter
        window_size = 11  # must be odd
        y_baseline_subtracted = medfilt(y_baseline_subtracted, kernel_size=window_size)

        # -------------------------------------------------------------------
        # ------------------------  1exp  -----------------------------------
        # -------------------------------------------------------------------
        def exp_func(t, A, tau):
            return A * np.exp(-t / tau)

        # Fit exponential from fit_st to fit_end
        fit_mask = (time >= fit_st) & (time <= fit_end)
        try:
            popt, _ = curve_fit(exp_func, time[fit_mask], y_baseline_subtracted[fit_mask],
                                p0=(np.max(y_baseline_subtracted), 5),bounds=([0, 0], [np.inf, np.inf]))
            A_fit, tau_fit = popt
        except Exception as e:
            print(f"\nFit failed for trace {trace_name}: {e}")
            popt = [np.nan, np.nan]
            A_fit, tau_fit = np.nan, np.nan # code could be optimize to yours only one of both options

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
        fit_plot_y = popt[0] * np.exp(-fit_plot_x / popt[1])
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

        # -------------------------------------------------------------------
        # ------------------------  1expY  -----------------------------------
        # -------------------------------------------------------------------
        # --- Exponential Fit with offset---
        def exp_funcY(t, A, tau, y0):
            return A * np.exp(-t / tau) + y0

        fit_mask = (time >= fit_st) & (time <= fit_end)
        try:
            popt, _ = curve_fit(exp_funcY, time[fit_mask], y_baseline_subtracted[fit_mask],
                                p0=(np.max(y_baseline_subtracted), 5, 0.0),bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
            A_fit, tau_fit, y0_fit = popt
        except Exception as e:
            print(f"\nFit failed for trace {trace_name}: {e}")
            popt = [np.nan, np.nan, np.nan]
            A_fit, tau_fit, y0_fit  = np.nan, np.nan, np.nan   # code could be optimize to yours only one of both options

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
        fit_plot_y = popt[0] * np.exp(-fit_plot_x / popt[1]) + popt[2]
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

        # -------------------------------------------------------------------
        # ------------------------  2exp  -----------------------------------
        # -------------------------------------------------------------------
        # --- Exponential Fit with 2exp ---
        def exp_func2(t, A, tau1, aRel, tau2):
            return A * (1 - aRel) * np.exp(-t / tau1) + A * aRel * np.exp(-t / tau2)

        fit_mask = (time >= fit_st) & (time <= fit_end)
        try:
            popt, _ = curve_fit(exp_func2, time[fit_mask], y_baseline_subtracted[fit_mask],
                                p0=(np.max(y_baseline_subtracted), 2, 0.5, 10),
                                bounds=([0, 0, 0, 0], [np.inf, np.inf, 1, np.inf]))
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

    print("Output of fit parameters... ", end="", flush=True)
    # -------------------------------------------------------------------
    # ------------------------  1exp  -----------------------------------
    # -------------------------------------------------------------------
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

        # Extract parameters values for each group and sequence
        for tmpStr in ['amplitude', 'tau']:
            group_c = results_df[(results_df['solution'] == 'c')][tmpStr].tolist()
            group_g = results_df[(results_df['solution'] == 'g')][tmpStr].tolist()

            group_c_seq1 = results_df[(results_df['solution'] == 'c') & (results_df['sequence'] == 1)][tmpStr].tolist()
            group_g_seq1 = results_df[(results_df['solution'] == 'g') & (results_df['sequence'] == 1)][tmpStr].tolist()

            group_c_seq2 = results_df[(results_df['solution'] == 'c') & (results_df['sequence'] == 2)][tmpStr].tolist()
            group_g_seq2 = results_df[(results_df['solution'] == 'g') & (results_df['sequence'] == 2)][tmpStr].tolist()

            # Call analyze_two_groups for each sequence
            myAna.analyze_two_groups(group_c, group_g, output_folder_parameterCompare_1exp, group_names=["c", "g"],
                                     title=tmpStr)
            myAna.analyze_two_groups(group_c_seq1, group_g_seq1, output_folder_parameterCompare_1exp,
                                     group_names=["c", "g"],
                                     title=f"{tmpStr} (Sequence 1)")
            myAna.analyze_two_groups(group_c_seq2, group_g_seq2, output_folder_parameterCompare_1exp,
                                     group_names=["c", "g"],
                                     title=f"{tmpStr} (Sequence 2)")

    # -------------------------------------------------------------------
    # ------------------------  1expY  -----------------------------------
    # -------------------------------------------------------------------
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

    # Extract parameters values for each group and sequence
    for tmpStr in ['amplitude','tau','y0']:
        group_c = results_df[(results_df['solution'] == 'c')][tmpStr].tolist()
        group_g = results_df[(results_df['solution'] == 'g')][tmpStr].tolist()

        group_c_seq1 = results_df[(results_df['solution'] == 'c') & (results_df['sequence'] == 1)][tmpStr].tolist()
        group_g_seq1 = results_df[(results_df['solution'] == 'g') & (results_df['sequence'] == 1)][tmpStr].tolist()

        group_c_seq2 = results_df[(results_df['solution'] == 'c') & (results_df['sequence'] == 2)][tmpStr].tolist()
        group_g_seq2 = results_df[(results_df['solution'] == 'g') & (results_df['sequence'] == 2)][tmpStr].tolist()

        # Call analyze_two_groups for each sequence
        myAna.analyze_two_groups(group_c, group_g, output_folder_parameterCompare_1expY, group_names=["c", "g"],
                           title=tmpStr)
        myAna.analyze_two_groups(group_c_seq1, group_g_seq1, output_folder_parameterCompare_1expY, group_names=["c", "g"],
                           title=f"{tmpStr} (Sequence 1)")
        myAna.analyze_two_groups(group_c_seq2, group_g_seq2, output_folder_parameterCompare_1expY, group_names=["c", "g"],
                           title=f"{tmpStr} (Sequence 2)")

    # -------------------------------------------------------------------
    # ------------------------  2exp  -----------------------------------
    # -------------------------------------------------------------------
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

    # Extract parameters values for each group and sequence
    for tmpStr in ['amplitude','tau1','aRel','tau2']:
        group_c = results_df[(results_df['solution'] == 'c')][tmpStr].tolist()
        group_g = results_df[(results_df['solution'] == 'g')][tmpStr].tolist()

        group_c_seq1 = results_df[(results_df['solution'] == 'c') & (results_df['sequence'] == 1)][tmpStr].tolist()
        group_g_seq1 = results_df[(results_df['solution'] == 'g') & (results_df['sequence'] == 1)][tmpStr].tolist()

        group_c_seq2 = results_df[(results_df['solution'] == 'c') & (results_df['sequence'] == 2)][tmpStr].tolist()
        group_g_seq2 = results_df[(results_df['solution'] == 'g') & (results_df['sequence'] == 2)][tmpStr].tolist()

        # Call analyze_two_groups for each sequence
        myAna.analyze_two_groups(group_c, group_g, output_folder_parameterCompare_2exp, group_names=["c", "g"],
                           title=tmpStr)
        myAna.analyze_two_groups(group_c_seq1, group_g_seq1, output_folder_parameterCompare_2exp, group_names=["c", "g"],
                           title=f"{tmpStr} (Sequence 1)")
        myAna.analyze_two_groups(group_c_seq2, group_g_seq2, output_folder_parameterCompare_2exp, group_names=["c", "g"],
                           title=f"{tmpStr} (Sequence 2)")

    print(" done!")

    plot_group_traces(
        time=time,
        traces_df=traces,
        traceName=traceName,
        solution=solution,
        sequence=sequence,
        output_folder=os.path.join(output_folder, "group_plots")
    )

if __name__ == '__main__':
    CmEval()

