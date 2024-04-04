import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


CSV_FILES = ['/home/dylanz/eecs592/results/evaluation/l2_1711844373.2195396']
COLORS = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]
LABELS = ['L2']
TRAINING_DIST = 0.9
FIGURE_NAME = 'l2-biased-skin-cancer.png'
RESULTS_DIR = '/home/dylanz'
TITLE = 'L2 Regularization Performance'
Y_LABEL = 'AUROC'
X_LABEL = 'P(Skin Cancer | White Skin) = P(Benign Lesion | Dark Skin)'
#X_LABEL = 'P(OA | Black Square) = P(Healthy | No Black Square) = 1.0'


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