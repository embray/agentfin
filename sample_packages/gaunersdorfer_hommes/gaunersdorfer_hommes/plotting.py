import numpy as np
import matplotlib.pyplot as plt


def plot_price(price):
    fig_p, ax_p = plt.subplots()
    ax_p.plot(np.arange(len(price)), price)
    plt.xlabel('Time')
    plt.ylabel('Price')
    return fig_p


def plot_returns(lret):
    fig_r, ax_r = plt.subplots()
    ax_r.plot(np.arange(len(lret)), lret)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    return fig_r
