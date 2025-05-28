import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, stats
from matplotlib.backends.backend_pdf import PdfPages

def analyze_two_groups(group_a, group_b, output_folder, group_names=None, title=None):
    if group_names is None:
        group_names = ["A", "B"]
    if title is None:
        title = "Two-group analysis"

    os.makedirs(output_folder, exist_ok=True)

    group_a = np.array(group_a)
    group_b = np.array(group_b)

    def get_stats(group):
        return {
            "mean": np.mean(group),
            "median": np.median(group),
            "SD": np.std(group, ddof=1),
            "SEM": np.std(group, ddof=1) / np.sqrt(len(group)),
            "IQR": np.percentile(group, 75) - np.percentile(group, 25),
            "n": len(group)
        }

    # Inferential statistics
    try:
        u_stat, p_mw = mannwhitneyu(group_a, group_b, alternative='two-sided')
    except Exception as e:
        u_stat, p_mw = np.nan, np.nan
        print("Mann-Whitney failed:", e)

    try:
        t_stat, p_t = ttest_ind(group_a, group_b, equal_var=False)
        df_t = len(group_a) + len(group_b) - 2
    except Exception as e:
        t_stat, p_t, df_t = np.nan, np.nan, np.nan
        print("t-test failed:", e)

    # PDF filename
    filename = f"two_group_{title.replace(' ', '_')}_{group_names[0]}_{group_names[1]}.pdf"
    pdf_path = os.path.join(output_folder, filename)

    # Create PDF
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        fig.suptitle(title, fontsize=16)

        # --- Table Column (Col 1) ---
        max_vals_display = 20
        n_vals_a = min(len(group_a), max_vals_display)
        n_vals_b = min(len(group_b), max_vals_display)
        n_vals = max(n_vals_a, n_vals_b)

        val_a = list(group_a[:n_vals]) + [np.nan] * (n_vals - n_vals_a)
        val_b = list(group_b[:n_vals]) + [np.nan] * (n_vals - n_vals_b)

        summary_labels = ["Mean", "Median", "SD", "SEM", "IQR", "n"]
        row_labels = [f"val{i+1}" for i in range(n_vals)] + summary_labels

        stats_a = get_stats(group_a)
        stats_b = get_stats(group_b)

        summary_vals_a = [stats_a[k] for k in ["mean", "median", "SD", "SEM", "IQR", "n"]]
        summary_vals_b = [stats_b[k] for k in ["mean", "median", "SD", "SEM", "IQR", "n"]]

        col1 = val_a + summary_vals_a
        col2 = val_b + summary_vals_b

        table = pd.DataFrame({group_names[0]: col1, group_names[1]: col2}, index=row_labels)

        axes[0].axis("off")
        table_obj = axes[0].table(cellText=table.values,
                                  rowLabels=table.index,
                                  colLabels=table.columns,
                                  loc="center",
                                  cellLoc='center')

        # Highlight summary rows by bolding the "Mean" row
        for row_idx, label in enumerate(row_labels):
            if label == "Mean":
                for col_idx in range(len(table.columns)):
                    try:
                        cell = table_obj[row_idx + 1, col_idx]  # +1 for header
                        cell.set_fontsize(10)
                        cell.set_linewidth(2.0)
                    except KeyError:
                        pass  # Skip missing cells

        axes[0].set_title("Descriptive Stats", fontsize=12)

        # --- Boxplot Column (Col 2) ---
        box = axes[1].boxplot(
            [group_a, group_b],
            labels=group_names,
            patch_artist=True,
            boxprops=dict(facecolor='none', color='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', color='black', alpha=0.5, markersize=4, linestyle='none')
        )

        # Individual data points
        positions = [1, 2]
        jitter = 0.08
        for i, group in enumerate([group_a, group_b]):
            x_vals = np.random.normal(positions[i], jitter, size=len(group))
            axes[1].scatter(x_vals, group, color='black', alpha=0.6, s=10, zorder=3)

        axes[1].set_title("Mann-Whitney U Test", fontsize=12)
        axes[1].set_ylabel("Value")

        mw_text = f"U = {u_stat:.2f}\nP = {p_mw:.4f}"
        axes[1].text(0.5, -0.15, mw_text, transform=axes[1].transAxes,
                     fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', edgecolor='black'))

        # --- Mean ± SEM Column (Col 3) ---
        means = [stats_a["mean"], stats_b["mean"]]
        sems = [stats_a["SEM"], stats_b["SEM"]]
        x_pos = [1, 2]
        axes[2].bar(x_pos, means, yerr=sems, align='center', alpha=0.6, capsize=10, color=['skyblue', 'salmon'])
        axes[2].scatter(np.ones(len(group_a)), group_a, color="black", alpha=0.5)
        axes[2].scatter(2 * np.ones(len(group_b)), group_b, color="black", alpha=0.5)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(group_names)
        axes[2].set_title("T-Test", fontsize=12)
        axes[2].set_ylabel("Value")

        t_text = f"t = {t_stat:.2f}\nP = {p_t:.4f}\ndf = {df_t}"
        axes[2].text(0.5, -0.15, t_text, transform=axes[2].transAxes,
                     fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', edgecolor='black'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved PDF to: {pdf_path}")
