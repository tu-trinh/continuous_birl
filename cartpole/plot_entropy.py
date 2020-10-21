import numpy as np
import matplotlib.pyplot as plt
import pickle    



def main():

    BETA = range(0,20)
    BETA = [b * 0.1 for b in BETA]

    beliefs = pickle.load( open( "choices/beliefs.pkl", "rb" ) )
    entropies = pickle.load( open( "choices/entropies.pkl", "rb" ) )

    b_mean = beliefs["mean"]
    e_mean = entropies["mean"]

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    color_counterfactuals = [255/255, 153./255, 0]
    color_noise = [102./255, 102./255, 102./255]
    color_classic = [179./255, 179./255, 179./255]
    colors = [color_classic, color_noise, color_counterfactuals]
    labels = ["classic", "noise", "counterfactuals"]

    for i in range(len(b_mean)):
        ax[0].plot(BETA, b_mean[i], color = colors[i], label=labels[i])
        ax[0].legend()
        ax[0].set_xlim(0, BETA[-1])
        ax[0].set_ylim([0, 1.0])
        ax[0].set_title("Belief wrt Beta")
        ax[0].set_xlabel("Beta")
        ax[0].set_ylabel("Belief")

        ax[1].plot(BETA, e_mean[i], color = colors[i], label=labels[i])
        ax[1].legend()
        ax[1].set_xlim(0, 1.0)
        # ax[1].set_ylim([0, 1.0])
        ax[1].set_title("Entropy wrt Beta")
        ax[1].set_xlabel("Beta")
        ax[1].set_ylabel("Entropy")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
