
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def save_fp_recall_plots(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby(["store", "camera_id"])

    for (store_id, cam_id), group in tqdm(grouped):
        if group["is_theft"].sum() == 0:
            continue  # Skip groups with no thefts

        y_true = group["is_theft"].values
        y_scores = group["probability"].values

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        false_positives = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            false_positives.append(fp)

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(recall[:-1], false_positives, drawstyle='steps-post')
        plt.title(f"Store {store_id} | Camera {cam_id}")
        plt.xlabel("Recall")
        plt.ylabel("False Positives")
        plt.grid(True)

        # Save
        filename = f"store_{store_id}_cam_{cam_id}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()

def display_fp_vs_recall_grid(df, max_cols=5):
    grouped = df.groupby(["store", "camera_id"])

    # Keep only stores with theft events 
    valid_groups = [
        ((store_id, cam_id), group)
        for (store_id, cam_id), group in grouped
        if group["is_theft"].sum() > 0
    ]

    total = len(valid_groups)
    ncols = max_cols
    nrows = (total + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for ax_idx, ((store_id, cam_id), group) in enumerate(valid_groups):
        y_true = group["is_theft"].values
        y_scores = group["probability"].values

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        false_positives = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            false_positives.append(fp)

        axes[ax_idx].plot(recall[:-1], false_positives, drawstyle='steps-post')
        axes[ax_idx].set_title(f"{store_id} | Cam {cam_id}", fontsize=8)
        axes[ax_idx].set_xlabel("Recall", fontsize=7)
        axes[ax_idx].set_ylabel("False Positives", fontsize=7)
        axes[ax_idx].tick_params(labelsize=6)
        axes[ax_idx].grid(True)

    for i in range(len(valid_groups), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


