"""
Example script for creating publication-ready violin plots.

This example demonstrates how to create violin plots with enhanced styling suitable
for publication. It includes:
- Custom styling with quartile lines
- Jitter points for better data visualization
- Publication-ready formatting (fonts, DPI, layout)

The script generates example data with different probability distributions and
creates a violin plot visualization.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.rcParams['text.usetex'] = False

def plot_violin(data, x, y, title, xlabel, output_path, color="lightblue", 
                figsize=(10, 6), font_scale=1.2, dpi=300, rotate_xlabels=0):
    """
    Create and save a publication-ready violin plot.
    
    Args:
        data: DataFrame containing the data to plot
        x: Column name for x-axis (categorical variable)
        y: Column name for y-axis (continuous variable)
        title: Plot title
        xlabel: X-axis label
        output_path: Path to save the plot
        color: Color scheme for the plot
        figsize: Figure size tuple (width, height)
        font_scale: Scale factor for font sizes
        dpi: Resolution for saved plot
        rotate_xlabels: Rotation angle for x-axis labels
    """
    # Set up publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.serif': ['Arial'],
        'text.usetex': False,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create enhanced violin plot
    vplot = sns.violinplot(
        x=x, 
        y=y, 
        inner='quartile', 
        data=data, 
        color=color,
        cut=0,
        order=data[x].sort_values(ascending=True).unique(),
        ax=ax,
        saturation=0.7,
        width=0.8
    )
    
        # Add subtle jitter points
    sns.stripplot(
        x=x,
        y=y,
        data=data,
        color=color,
        alpha=0.05,
        size=0.5,
        jitter=0.15,
        ax=ax,
        order=data[x].sort_values(ascending=True).unique()
    )
    
    # Enhance violin plot appearance
    for violin in vplot.collections:
        violin.set_alpha(0.8)
        violin.set_edgecolor('black')
        violin.set_linewidth(0.8)
    
    # Customize quartile lines
    for l in ax.lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)
    
    # Enhance grid
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    
    # Set labels and title with LaTeX formatting
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel('True Class Probability', labelpad=10)
    if title:
        ax.set_title(title, pad=15)
    
    # Rotate x-labels if specified
    if rotate_xlabels:
        plt.xticks(rotation=rotate_xlabels)
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Create different distributions for each category
categories = ['A', 'B', 'C', 'D']
n_samples = 1000
data_dict = {
    'category': [],
    'probability': []
}

# Generate data with different distributions for each category
for cat in categories:
    if cat == 'A':
        # Normal distribution centered at 0.7
        probs = np.random.normal(0.7, 0.15, n_samples)
    elif cat == 'B':
        # Bimodal distribution
        probs = np.concatenate([
            np.random.normal(0.3, 0.1, n_samples//2),
            np.random.normal(0.8, 0.1, n_samples//2)
        ])
    elif cat == 'C':
        # Skewed distribution
        probs = np.random.beta(5, 2, n_samples)
    else:
        # Uniform distribution
        probs = np.random.uniform(0.1, 0.9, n_samples)
    
    # Clip probabilities to [0,1] range
    probs = np.clip(probs, 0, 1)
    
    data_dict['category'].extend([cat] * n_samples)
    data_dict['probability'].extend(probs)

# Create DataFrame
df = pd.DataFrame(data_dict)

# Create the plot
plot_violin(
    data=df,
    x='category',
    y='probability',
    title='Distribution of Probabilities by Category',
    xlabel='Category',
    output_path=Path('violin_plot.png'),
    color='lightblue',
    figsize=(8, 6),
    font_scale=1.2,
    dpi=300
)

# Print summary statistics
print("\nSummary statistics for each category:")
print(df.groupby('category')['probability'].describe())