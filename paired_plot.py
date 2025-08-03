def paired_boxplot(group1_vals, group2_vals, group_label, parameter_name, output_folder):
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    os.makedirs(output_folder, exist_ok=True)

    paired_df = pd.DataFrame({
        "Stim1": group1_vals,
        "Stim2": group2_vals
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    # Define x-axis positions (closer together)
    box1_x = 0.0
    dots1_x = 0.6
    dots2_x = 1.0
    box2_x = 1.6

    # Draw paired lines and dots
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

    # Boxplots — placed manually
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

    # Customize x-axis
    ax.set_xticks([box1_x, dots1_x, dots2_x, box2_x])
    ax.set_xticklabels(["Box 1", "Stim1", "Stim2", "Box 2"])
    ax.set_xlim(-0.5, 2.1)

    ax.set_title(f"{parameter_name} — Group {group_label}", fontsize=13)
    ax.set_ylabel(parameter_name)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    filename = f"paired_plot_{parameter_name}_group_{group_label}.pdf"
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()
