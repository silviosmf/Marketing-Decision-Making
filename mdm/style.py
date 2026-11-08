"""Figure style for the Journal of Marketing Analytics: grayscale, Arial/Helvetica, legible in black and white."""
import logging
import os

import matplotlib
import matplotlib.pyplot as plt

GRAYS = ['#000000', '#555555', '#999999', '#cccccc']          # dark to light
HATCHES = ['', '///', '...', 'xx']                             # second cue besides the gray level


def use():
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # Arial missing on some systems
    matplotlib.rcParams.update({
        'font.family': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.edgecolor': '#000000', 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'hatch.linewidth': 0.6,
    })


def save(fig, name, folder='figures'):
    """Save PNG (300 dpi), PDF and SVG; returns the PNG path."""
    os.makedirs(folder, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(folder, f'{name}.{ext}'), facecolor='white')
    return os.path.join(folder, f'{name}.png')
