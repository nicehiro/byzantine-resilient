import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


root_path = "logs"
paths = ["baseline"]


for path in paths:
    logdir = os.path.join(root_path, path)
    logs_path = os.listdir(logdir)
    logs_list = []
    for log_path in logs_path:
        log = np.loadtxt(os.path.join(logdir, log_path), delimiter=",")
        logs_list.append(log)
    logs = pd.DataFrame.from_dict(dict(zip(paths, logs_list)))
    # get min acc log
    min_acc = logs.min(axis=1)
    # draw min acc
    min_acc.plot()

plt.savefig("acc.png")
