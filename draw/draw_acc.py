import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


attack = "gaussian"
root_path = f"logs/mnist/{attack}/by-5/"
pars = [
    "average",
    "bridge",
    "median",
    "krum",
    "bulyan",
    "zeno",
    # "mozi",
    "qc",
]
paths = ["baseline"] + [f"{attack}-{par}" for par in pars]

final_acc = []

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
    label = "baseline" if i == 0 else pars[i - 1]
    # save final acc
    final_acc.append(min_acc[len(min_acc) - 1])
    min_acc.plot(label=label)

np.savetxt("acc.csv", np.array(final_acc))

plt.legend()
plt.savefig("mnist-acc.png")
