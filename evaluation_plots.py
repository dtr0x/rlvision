import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__ == "__main__":
    data = np.load("recall_planes.npz")

    recalls = data['recalls']
    mean_actions = data['mean_actions']

    zero_recall = np.nonzero(recall)

    best_model_idx = np.argmin(recall[np.nonzero(recall)])



    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
    ax1.plot(recall)

    ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
    ax2.plot(mean_actions)

    ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns
    ax3.plot(recall)

    plt.show()