import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set(font="Times New Roman")

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


# matplotlib.use("Agg")
attack = "empire"
root_path = f"logs/mnist-centralized/{attack}/1/"
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
paths = [f"{attack}-{par}" for par in pars]

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

final_acc = []

fig, axs = plt.subplots(1, 1, figsize=[6, 6])

# baseline
axs.axhline(y=0.925, color="r", linestyle="-")

for i, path in enumerate(paths):
    if path != "baseline":
        logdir = os.path.join(root_path, path)
    else:
        logdir = os.path.join("logs/mnist/", path)
    logs_path = os.listdir(logdir)
    logs_list = []
    non_zero_logs = []
    for log_path in logs_path:
        log = np.loadtxt(os.path.join(logdir, log_path), delimiter=",")
        if log.size != 0:
            logs_list.append(log)
            non_zero_logs.append(log_path)
    logs = pd.DataFrame.from_dict(dict(zip(non_zero_logs, logs_list)))
    # get min acc log
    min_acc = logs.min(axis=1)
    # draw min acc
    label = pars[i]
    # save final acc
    final_acc.append(min_acc[len(min_acc) - 1])
    axs.plot(
        min_acc, label=label, marker=markers[i], alpha=alphas[i], color=dark_color[i]
    )

np.savetxt(f"{root_path}acc.csv", np.array(final_acc))

axs.set_ylabel("Accuracy/%")
axs.set_xlabel("Epochs")
fig.legend(ncol=3, loc="lower center")
plt.subplots_adjust(left=0.13, right=0.95, bottom=0.25, top=0.95, wspace=0)
# plt.savefig(f"{root_path}mnist-acc.eps", format="eps")
plt.savefig(f"{root_path}mnist-acc.png")
# plt.show()
