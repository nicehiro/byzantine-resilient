import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


root_path = "logs/mnist"
paths = ["baseline", "max-average", "max-median", "max-qc"]


for path in paths:
    logdir = os.path.join(root_path, path)
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
    min_acc.plot()

plt.savefig("mnist-acc.png")
