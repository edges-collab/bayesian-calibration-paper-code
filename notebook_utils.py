import matplotlib.pyplot as plt

import edges_cal
import edges_io
import read_acq
import edges_analysis
import numpy as np
import astropy as ap

single_width = 3.377  # inches
double_width = 7.03   # inches
page_height = 9.43869 # inches
full_page = (double_width, page_height)
half_page = (single_width, page_height*single_width / double_width)

def setup_mpl():
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size']= 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['hatch.linewidth'] = 0.7


def print_versions():
    print("Numpy Version: ", np.__version__)
    print("Astropy Version: ", ap.__version__)
    print("read_acq Version: ", read_acq.__version__)
    print("edges-io Version: ", edges_io.__version__)
    print("edges-cal Version: ", edges_cal.__version__)
    print("edges-analysis Version: ", edges_analysis.__version__)

