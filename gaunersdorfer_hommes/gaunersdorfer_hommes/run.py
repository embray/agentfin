from .gaunersdorfer_hommes import gaunersdorfer_hommes
from .plotting import plot_price, plot_returns


def run(prices_filename=None, returns_filename=None):
    p, lret, ret, ghret = gaunersdorfer_hommes()

    fig = plot_price(p)
    if prices_filename:
        fig.savefig(prices_filename)

    fig = plot_returns(lret)

    if returns_filename:
        fig.savefig(returns_filename)
