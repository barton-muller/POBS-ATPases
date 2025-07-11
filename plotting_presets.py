import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

SAVE_FIGS = True

# Optional: Apply Seaborn base style for spacing and aesthetics
sns.set_style("white")  # "whitegrid" also fine if grid desired

# Custom color cycle
from matplotlib.colors import ListedColormap

# Your named colors
nicecolors = { #hex or mpl color: humman name
    "royalblue":       "royalblue",
    "darkorange":      "darkorange",
    "#4CAB6F":         "green",
    "#D16FFF":         "pink",
    "#5ECCF3":         "lightblue",
    "#F14124":         "red",
    "#611FAD":         "purple",
    "teal":            "teal"
}
cnames = {v: k for k, v in nicecolors.items()}


# Create colormap from keys (the real color values)
nice_cmap = ListedColormap(list(nicecolors.keys()), name="nice_cmap")

# Update rcParams
mpl.rcParams.update({
    # === Figure ===
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'figure.facecolor': 'white',

    # === Axes ===
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'axes.edgecolor': 'black',
    'axes.spines.top': False,   # <- hide top spine
    'axes.spines.right': False, # <- hide right spine
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.prop_cycle': mpl.cycler(color=nicecolors),

    # === Ticks ===
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,

    # Tick locators (soft default of ~5 major ticks)
    # 'axes.autolimit_mode': 'round_numbers',
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,

    # === Lines and markers ===
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'lines.markeredgewidth': 0.8,

    # === Font ===
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],

    # === Legend ===
    'legend.fontsize': 12,
    'legend.frameon': False,
    'legend.loc': 'best',

    # === Savefig ===
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,

    # === Colormap ===
    'image.cmap': 'plasma'
})

# Then apply seaborn with custom settings (optional)
sns.set_style("white", {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.bottom": True,
    "ytick.left": True
})

import os
# Save original savefig function
_original_savefig = plt.savefig

def savefig_with_folder(fname, *args, folder="figs", **kwargs):
    # If fname is not absolute path, prepend folder
    if SAVE_FIGS:
        if not os.path.isabs(fname):
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, fname)
        return _original_savefig(fname, *args, **kwargs)
    else:
        print('Currently not saving figures')

# Replace plt.savefig with our wrapper
plt.savefig = savefig_with_folder

# fitting functions and their label, custom can be made in simmilar fashion
linear_fit = lambda x, a, b: a*x + b
linear_label = lambda a, b: f"$y = {a:.2f}x+ {b:.2f}$"

quad_fit = lambda x, a, b, c: a*x**2 + b*x +c
quad_label = lambda a, b, c: f"$y = {a:.2f}x^2 + {b:.2f}x+c$"

exp_fit = lambda x, a, b: a* np.exp( b * x)
exp_label = lambda a, b: f"$y = {a:.2f} e^{{{b:.2f} x}}$"
def plot_fit(ax, x, y, fit_func, label=None, **kwargs):
    """
    Fits a curve to the data and plots the fit on the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        fit_func (callable): The function to use for fitting.
        color (str, optional): The color of the fit line. Defaults to None.
        linestyle (str, optional): The linestyle of the fit line. Defaults to None.
        label (str or callable, optional): The label for the fit line. If callable,
          it will be called with the fit parameters. Defaults to None.

    Returns:
        tuple: A tuple containing the fit parameters, covariance matrix, and R^2 value.
    """
    # Fit curve
    popt, popc = curve_fit(fit_func, x, y)
    
    if callable(label):
        label = label(*popt)
    spread = max(x) - min(x)
    xfit = np.linspace(min(x) - spread, max(x) + spread, 1000)
    y_fit = fit_func(xfit, *popt)
    label = f'{label}' if label else None
    sns.lineplot(ax=ax, x=xfit, y=y_fit, label=label, **kwargs)

    # Compute R^2
    ss_res = np.sum((y - fit_func(x, *popt)) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    R2 = 1 - (ss_res / ss_tot)
    # print(f'Fit parameters: {popt}', f' R^2: {R2}')
    return popt, popc, R2