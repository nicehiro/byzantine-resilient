import os
import re

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# matplotlib.use("Agg")
sns.set(font="Times New Roman")

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.use("Agg")

pars = [
    "average",
    "bridge",
    "median",
    "krum",
    "bulyan",
    "zeno",
    "mozi",
    "qc",
]

pars_label = {
    "Average",
    "BRIDGE",
    "Dmedian",
    "DKrum",
    "DBulyan",
    "DZeno",
    "UBAR",
    "QC",
}

attacks = [
    "max",
    "gaussian",
    "hidden",
    "litter",
    "empire",
]

attacks_label = [
    "Max",
    "Gaussian",
    "Hidden",
    "A Litter is Enough",
    "Fall of Empires",
]

root_path = "logs/"

# csv files
paths = [
    [
        "mnist/max-04-05.csv",
        "mnist/gaussian-04-05.csv",
        "mnist/hidden-04-05.csv",
        "mnist/litter-04-05.csv",
        "mnist/empire-04-05.csv",
    ],
    [
        "mnist/qc-0.2-0.1.csv",
        "mnist/qc-0.2-0.3.csv",
        "mnist/qc-0.2-0.5.csv",
        "mnist/qc-0.4-0.1.csv",
        "mnist/qc-0.4-0.3.csv",
        "mnist/qc-0.4-0.5.csv",
        "mnist/qc-0.6-0.1.csv",
        "mnist/qc-0.6-0.3.csv",
        "mnist/qc-0.6-0.5.csv",
    ]
    # [
    #     "cifar/max-04-05.csv",
    #     "cifar/gaussian-04-05.csv",
    #     "cifar/hidden-04-05.csv",
    #     "cifar/litter-04-05.csv"
    #     "cifar/empire-04-05.csv",
    # ],
]

markers = [""] * 7 + ["^"]
alphas = [0.5] * 7 + [1.0]
dark_color = [
    # "#B71C1C",
    "#4A148C",
    "#B388FF",
    "#1565C0",
    "#00695C",
    "#2E7D32",
    "#9E9D24",
    "#EF6C00",
    "blue",
]

def plot1():
    fig, axs = plt.subplots(2, 5, figsize=[10, 5], sharex=True, sharey=False)

    for i in range(len(paths)):
        for j in range(len(paths[i])):
            csv_path = os.path.join(root_path, paths[i][j])
            d = pd.read_csv(csv_path, index_col=0)
            # axs[i][j].plot(d, color=tuple(dark_color), alpha=alphas, marker=markers)
            r = d.plot(ax=axs[i,j], color=dark_color, legend=False, xlim=(0, 50), ylim=(0,1))

    for i in range(2):
        for j in range(5):
            if i == 1:
                axs[i][j].set_yticklabels([])
            elif j == 0:
                axs[i][j].set_xticklabels([])
            else:
                axs[i][j].set_xticklabels([])
                axs[i][j].set_yticklabels([])
    axs[1][0].set_xticklabels([0, 25, 50])
    axs[1][0].set_yticklabels([0.00, 0.25, 0.50, 0.75, 1.00])

    # axs[1][2].set_xlabel("Epochs")
    fig.text(0.5, 0.14, 'Epochs', ha='center')
    fig.text(0.01, 0.6, 'Accuracy/%', va='center', rotation='vertical')
    fig.legend(labels=pars, ncol=4, loc="lower center")
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.96, wspace=0.13, hspace=0.13)
    # plt.savefig(f"{root_path}mnist-acc.eps", format="eps")
    # plt.savefig(f"{root_path}mnist-acc.png")
    plt.show()


def plot2(path):
    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)
    fig = plt.figure(figsize=(15, 8))

    gs = GridSpec(2, 6, figure=fig)
    axs = []
    axs.append(fig.add_subplot(gs[0, 1:3]))
    axs.append(fig.add_subplot(gs[0, 3:5]))
    axs.append(fig.add_subplot(gs[1, 0:2]))
    axs.append(fig.add_subplot(gs[1, 2:4]))
    axs.append(fig.add_subplot(gs[1, 4:6]))

    for i in range(len(path)):
        csv_path = os.path.join(root_path, path[i])
        d = pd.read_csv(csv_path, index_col=0)
        # axs[i][j].plot(d, color=tuple(dark_color), alpha=alphas, marker=markers)
        r = d.plot(ax=axs[i], color=dark_color, legend=False, xlim=(0, 50), ylim=(0,1))
        axs[i].set_ylabel("Accuracy %")
        axs[i].set_xlabel("Epochs")
        axs[i].set_title(attacks_label[i])

    plt.subplots_adjust(left=0.065, right=0.98, bottom=0.17, top=0.95, wspace=0.54, hspace=0.36)
    fig.legend(labels=pars_label, ncol=4, loc="lower center")
    plt.savefig(f"{root_path}mnist-acc.eps", format="eps")
    plt.show()


def plot3(path):
    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)
    fig = plt.figure(figsize=(10, 9))

    gs = GridSpec(3, 3, figure=fig)
    axs = []
    axs.append(fig.add_subplot(gs[0, 0]))
    axs.append(fig.add_subplot(gs[0, 1]))
    axs.append(fig.add_subplot(gs[0, 2]))
    axs.append(fig.add_subplot(gs[1, 0]))
    axs.append(fig.add_subplot(gs[1, 1]))
    axs.append(fig.add_subplot(gs[1, 2]))
    axs.append(fig.add_subplot(gs[2, 0]))
    axs.append(fig.add_subplot(gs[2, 1]))
    axs.append(fig.add_subplot(gs[2, 2]))

    for i in range(len(path)):
        _, cr, br = re.split("[-]", path[i][:-4])
        csv_path = os.path.join(root_path, path[i])
        d = pd.read_csv(csv_path, index_col=0)
        # axs[i][j].plot(d, color=tuple(dark_color), alpha=alphas, marker=markers)
        r = d.plot(ax=axs[i], color=dark_color, legend=False, xlim=(0, 50), ylim=(0,1))
        axs[i].set_ylabel("Accuracy %")
        axs[i].set_xlabel("Epochs")
        axs[i].set_title(f"cr: {cr} br: {br}")

    plt.subplots_adjust(left=0.065, right=0.98, bottom=0.12, top=0.95, wspace=0.29, hspace=0.39)
    fig.legend(labels=attacks_label, ncol=5, loc="lower center")
    plt.savefig(f"{root_path}qc-mnist-acc.eps", format="eps")
    plt.show()


if __name__ == '__main__':
    # plot2(paths[0])
    plot3(paths[1])
