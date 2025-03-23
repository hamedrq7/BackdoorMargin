import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib

W = 10
H = 0.4 * W

matplotlib.rc('text')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

FONT_SIZE = 15
MARKER_SIZE = 15
LINEWIDTH = 2
SAMPLE_SIZE = 3

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

poisoned_poisoned_margins = np.load('C:/Users/hamed/Downloads/margins_npy/poisoned/margins - poisoned.npy')
poisoned_clean_margins = np.load('C:/Users/hamed/Downloads/margins_npy/poisoned/margins - clean.npy')

benign_poisoned_margins = np.load('C:/Users/hamed/Downloads/margins_npy/benign/margins - poisoned.npy')
benign_clean_margins = np.load('C:/Users/hamed/Downloads/margins_npy/benign/margins - clean.npy')

def swarmplot_list(margin_list, labels, alpha=0.3, jitter=0.1, colors=None, s=SAMPLE_SIZE, discard_ratio=0):
    """
    Plot multiple arrays of margins on the same figure.
    
    Args:
    - margin_list: List of numpy arrays, where each array contains margin data to be plotted.
    - labels: List of labels corresponding to each margin array.
    - alpha: Transparency level for scatter points.
    - jitter: Jitter added to the scatter points for better visualization.
    - colors: List of colors for each margin array.
    - s: Size of scatter points.
    - discard_ratio: Ratio of points to discard for each margin array.
    """
    plt.figure(figsize=(W, H))
    
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple']  # Default colors
    
    max_median = 0
    for i, margins in enumerate(margin_list):
        idx = np.tile(np.arange(margins.shape[0]), (margins.shape[1], 1)).T
        # Plot median line
        plt.plot(idx[:, 0], np.median(margins, axis=1), '.-', linewidth=LINEWIDTH, 
                 color=colors[i % len(colors)], markersize=MARKER_SIZE, label=labels[i])
        
        # Update max_median for axis limits
        max_median = max(max_median, np.median(margins, axis=1).max())
        
        # Plot scatter points
        margins = margins[:, :int(margins.shape[1] * (1 - discard_ratio))]
        idx = idx[:, :int(margins.shape[1])]
        plt.scatter(idx[:] + jitter * np.random.randn(*margins.shape), margins[:], 
                    alpha=alpha, color=colors[i % len(colors)], s=s)
    
    new_xticks = ['Low', 'Frequency', 'High']
    plt.xticks([-1, 10, 21], new_xticks, rotation=0, horizontalalignment='center')
    plt.axis([-1, 21, 0, max_median * 1.3])
    plt.ylabel('Margin')
    plt.legend()
    plt.show()


swarmplot_list([poisoned_poisoned_margins, poisoned_clean_margins, benign_poisoned_margins, benign_clean_margins], ['posined_posined', 'posined_clean', 'benign_posined', 'benign_clean'])

