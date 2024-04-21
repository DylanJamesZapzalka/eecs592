import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


CSV_FILES = ['/home/dylanz/eecs592/results/evaluation/l2_0.95',
            '/home/dylanz/eecs592/results/evaluation/mmd_0.95',
            '/home/dylanz/eecs592/results/evaluation/gdro_0.95',
            '/home/dylanz/eecs592/results/evaluation/tipmi-cf_0.95'
            ]

COLORS = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]
LABELS = ['L2', 'MMD', 'GDRO', 'TIPMI']
TRAINING_DIST = 0.95
FIGURE_NAME = 'skin-cancer-0.95.png'
RESULTS_DIR = '/home/dylanz'
TITLE = 'Skin Cancer Distribution Shift Performance'
Y_LABEL = 'AUROC'
X_LABEL = 'P(Skin Cancer | Light Skin) = P(Benign Lesion | Dark Skin)'



for i in range(len(CSV_FILES)):
    # Load dataframe
    df = pd.read_csv(CSV_FILES[i])

    # Get mean and range for each test distribution
    means_df = df.mean(axis=0)
    test_distributions = means_df.index.to_numpy(dtype=np.float32)
    means = means_df.to_numpy()
    std_errs = df.std() / np.sqrt(df.shape[0])

    # Plot mean values with error bars
    plt.plot(test_distributions, means, label=LABELS[i], color=COLORS[i])
    plt.errorbar(test_distributions, means, yerr=std_errs, fmt='o', color=COLORS[i], capsize=10)

# Draw dashed vertical line
plt.axvline(x=TRAINING_DIST, linestyle='--', color='grey')

plt.legend()

# Create labels and title
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.title(TITLE)

# Save the figure
plt.savefig(os.path.join(RESULTS_DIR, FIGURE_NAME))