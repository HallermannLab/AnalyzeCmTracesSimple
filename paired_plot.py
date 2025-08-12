import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon, ttest_rel

def paired_boxplot(group1_vals, group2_vals, group_label, parameter_name, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    paired_df = pd.DataFrame({
        "Stim1": group1_vals,
        "Stim2": group2_vals
    })

    # Statistische Tests
    try:
        stat_wilcoxon, p_wilcoxon = wilcoxon(group1_vals, group2_vals)
    except Exception:
        stat_wilcoxon, p_wilcoxon = np.nan, np.nan

    try:
        t_stat, p_t = ttest_rel(group1_vals, group2_vals)
    except Exception:
        t_stat, p_t = np.nan, np.nan

    # Tabelleninhalte vorbereiten
    max_vals_display = 20
    n_vals = min(len(group1_vals), max_vals_display)

    val1 = list(group1_vals[:n_vals]) + [np.nan] * (max_vals_display - n_vals)
    val2 = list(group2_vals[:n_vals]) + [np.nan] * (max_vals_display - n_vals)

    def get_stats(group):
        return {
            "mean": np.mean(group),
            "median": np.median(group),
            "SD": np.std(group, ddof=1),
            "SEM": np.std(group, ddof=1) / np.sqrt(len(group)),
            "IQR": np.percentile(group, 75) - np.percentile(group, 25),
            "n": len(group)
        }

    stats1 = get_stats(group1_vals)
    stats2 = get_stats(group2_vals)

    summary_labels = ["Mean", "Median", "SD", "SEM", "IQR", "n"]
    summary_vals1 = [stats1[k] for k in ["mean", "median", "SD", "SEM", "IQR", "n"]]
    summary_vals2 = [stats2[k] for k in ["mean", "median", "SD", "SEM", "IQR", "n"]]

    row_labels = [f"val{i+1}" for i in range(max_vals_display)] + summary_labels

    table1_df = pd.DataFrame({
        "Stim1": val1 + summary_vals1,
        "Stim2": val2 + summary_vals2
    }, index=row_labels)

    test_table = pd.DataFrame({
        'Test': ["Wilcoxon", "T-test"],
        'Stat': [f"{stat_wilcoxon:.2f}" if not np.isnan(stat_wilcoxon) else "–",
                 f"{t_stat:.2f}" if not np.isnan(t_stat) else "–"],
        'p-value': [f"{p_wilcoxon:.4f}" if not np.isnan(p_wilcoxon) else "–",
                    f"{p_t:.4f}" if not np.isnan(p_t) else "–"]
    })

    # Plot vorbereiten
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2])

    ax_table1 = fig.add_subplot(gs[0])
    ax_table1.axis('off')
    ax_table1.set_title("Deskriptive Statistik")
    table_obj1 = ax_table1.table(cellText=table1_df.values,
                                 rowLabels=table1_df.index,
                                 colLabels=table1_df.columns,
                                 loc="center",
                                 cellLoc='center')
    table_obj1.auto_set_font_size(False)
    table_obj1.set_fontsize(8)

    ax_table2 = fig.add_subplot(gs[1])
    ax_table2.axis('off')
    ax_table2.set_title("Statistische Tests")
    table_obj2 = ax_table2.table(cellText=test_table.values,
                                 colLabels=test_table.columns,
                                 loc="center",
                                 cellLoc='center')
    table_obj2.auto_set_font_size(False)
    table_obj2.set_fontsize(10)

    # Boxplot & Linien
    ax = fig.add_subplot(gs[2])

    box1_x = 0.0
    dots1_x = 0.6
    dots2_x = 1.0
    box2_x = 1.6

    for i in range(len(paired_df)):
        ax.plot(
            [dots1_x, dots2_x],
            [paired_df.iloc[i]["Stim1"], paired_df.iloc[i]["Stim2"]],
            marker='o',
            color='black',
            alpha=1,
            linewidth=1,
            markersize=6,
            zorder=2
        )

    boxplot_width = 0.25
    ax.boxplot(
        group1_vals,
        positions=[box1_x],
        widths=boxplot_width,
        patch_artist=True,
        boxprops=dict(facecolor='skyblue', alpha=0.5),
        medianprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=3, color='black'),
        zorder=1
    )
    ax.boxplot(
        group2_vals,
        positions=[box2_x],
        widths=boxplot_width,
        patch_artist=True,
        boxprops=dict(facecolor='hotpink', alpha=0.5),
        medianprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=3, color='black'),
        zorder=1
    )

    ax.set_xticks([box1_x, dots1_x, dots2_x, box2_x])
    ax.set_xticklabels(["Box 1", "Stim1", "Stim2", "Box 2"])
    ax.set_xlim(-0.5, 2.1)
    ax.set_title(f"{parameter_name} — Group {group_label}", fontsize=13)
    ax.set_ylabel(parameter_name)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Signifikanz-Sternchen
    if p_wilcoxon < 0.001:
        sig_label = "***"
    elif p_wilcoxon < 0.01:
        sig_label = "**"
    elif p_wilcoxon < 0.05:
        sig_label = "*"
    else:
        sig_label = "n.s."

    # Höhe des Sternchens: fix vom oberen Achsenrand
    y_max = max(max(group1_vals), max(group2_vals))
    y_min = min(min(group1_vals), min(group2_vals))
    offset = 0.07 * (y_max - y_min)
    y_sig = y_max - offset

    ax.text((dots1_x + dots2_x) / 2, y_sig, sig_label,
            ha='center', va='top', fontsize=14, weight='normal')

    plt.tight_layout()

    filename = f"paired_plot_{parameter_name}_group_{group_label}.pdf"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.close()

    print(f"Saved paired plot for group: {group_label} -> {filepath}", flush=True)
