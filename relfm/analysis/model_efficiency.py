"""Plots avg inference time vs. model size."""
import numpy as np
import matplotlib.pyplot as plt

from relfm.utils.visualize import set_latex_fonts


models = ["R2D2", "C-3PO ($C_{4}$)", "C-3PO ($C_{8}$)", "C-3PO ($SO(2)$)"]
avg_inf_time = [0.022, 0.084, 0.228, 0.192]
num_of_params = [484387, 1422403, 2827331, 1462339]


colors = ["blue", "gold", "green", "red"]
markers = ["o", "D", "X", "s"]

if __name__ == "__main__":
    set_latex_fonts(show_sample=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for i in range(len(models)):
        ax.scatter(num_of_params[i], avg_inf_time[i], label=models[i], s=100, color=colors[i], marker=markers[i])

    ax.legend(fontsize=17)
    ax.grid(alpha=0.5)

    ax.set_xticks(ax.get_xticks(), minor=True)
    ax.set_xlabel("Number of trainable parameters", fontsize=17)
    ax.set_ylabel("Avg inference time per image (s)", fontsize=17)
    ax.set_title("Model Efficiency", fontsize=18)

    plt.savefig("./Figures/model_efficiency.pdf", bbox_inches="tight")