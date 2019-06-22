from .gaunersdorfer_hommes import gaunersdorfer_hommes
from .plotting import plot_price, plot_returns

p, lret, ret, ghret = gaunersdorfer_hommes()

plot_price(p)
plot_returns(lret)
