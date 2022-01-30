import os
import re

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font="Times New Roman")

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# matplotlib.use("Agg")

def line_plot():
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    methods = ["Average", "BRIDGE", "DKrum", "Dmedian", "DBulyan", "DZeno", "DUBAR", "CA-PAR"]
    mnist_agg_times = [0.0026, 0.0129, 0.0263, 0.0047, 0.0071, 4.1740, 0.5605, 0.00005]
    cifar_agg_times = [0.0189, 0.0270, 0.2279, 0.0218, 0.3305, 5.3735, 2.6980, 0.0295]

    ax.plot(methods, mnist_agg_times, label='MNIST')
    ax.plot(methods, cifar_agg_times, label='CIFAR10')
    ax2.plot(methods, mnist_agg_times, label='MNIST')
    ax2.plot(methods, cifar_agg_times, label='CIFAR10')

    ax.set_ylim(4, 5.5)
    ax2.set_ylim(0, 0.5)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.xticks(rotation=45)

    ax.legend()
    plt.show()


def bar_plot():
    methods = ["Average", "BRIDGE", "DKrum", "Dmedian", "DBulyan", "DZeno", "DUBAR", "CA-PAR"]
    N = 8
    width = 0.35
    ind = np.arange(N)
    mnist_agg_times = [0.0026, 0.0129, 0.0263, 0.0047, 0.0071, 4.1740, 0.5605, 0.00005]
    cifar_agg_times = [0.0189, 0.0270, 0.2279, 0.0218, 0.3305, 5.3735, 2.6980, 0.0295]

    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, mnist_agg_times, width, color='r')
    rects2 = ax.bar(ind + width, cifar_agg_times, width, color='y')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%f' % height,
                    ha='center', va='bottom', 
                    rotation=45)

    autolabel(rects1)
    autolabel(rects2)

    plt.bar(ind, mnist_agg_times, width, color='green', label="MNIST")
    plt.bar(ind+width, cifar_agg_times, width, color='blue', label="CIFAR10")
    plt.xlabel("PARs")
    plt.ylabel("Computation Time(s)")
    plt.xticks(ind + width / 2, methods, rotation=45)
    plt.legend()
    plt.show()

def bar_plot2():
    brs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N = 9
    width = 0.35
    ind = np.arange(N)

    hidden_epochs = [8, 7, 13, 18, 14, 25, 24, 23, 21]
    gaussian_epochs = [4, 5, 9, 12, 20, 26, 41, 58, 75]
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, gaussian_epochs, width, color='r')
    rects2 = ax.bar(ind + width, hidden_epochs, width, color='y')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1 + height,
                    '%d' % height,
                    ha='center', va='bottom', 
                    rotation=0)

    autolabel(rects1)
    autolabel(rects2)

    plt.bar(ind, gaussian_epochs, width, color='green', label="Gaussian")
    plt.bar(ind+width, hidden_epochs, width, color='blue', label="Hidden")
    plt.xlabel("Byzantine Ratios")
    plt.ylabel("Epochs to reach the accuracy of 0.8")
    plt.xticks(ind + width / 2, brs, rotation=45)
    plt.legend()
    plt.subplots_adjust(bottom=0.157)
    plt.show()
    plt.savefig("brs_0.8.eps", format="eps")

if __name__ == '__main__':
    bar_plot2()