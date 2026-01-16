import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path 

def plot_tissue_attribution(
    input_file: str,
    output_dir: str,
    compartment: str = 'Total',
    figsize: tuple = (10, 6),
    colors: list = ['#00A087FF', '#91D1C2FF'],
    font_scale: float = 1.2,
    dpi: int = 300,
    plot_metric: str = 'ratio_tissue_vs_black_attr'
) -> None:
    """
    Plot tissue attribution analysis from integrated gradients data.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save output plots
        compartment: Compartment to filter for
        figsize: Figure size as (width, height)
        colors: List of two colors for the bars
        font_scale: Scale factor for font sizes
        dpi: DPI for PNG output
    """
    # Read and filter data
    df = pd.read_csv(input_file)
    # df = df[df['compartment'] == compartment]
    
    # Replace data type names
    df['data_type'] = df['data_type'].replace({
        'random_image': 'Random Imputation',
        'black': 'Black'
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=font_scale)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create paired barplot
    bar_plot = sns.barplot(
        data=df,
        x='threshold',
        y=plot_metric,
        hue='data_type',
        palette=colors,
        errorbar=None,
        ax=ax
    )
    
    # Calculate and add delta indicators
    thresholds = df['threshold'].unique()
    for thresh in thresholds:
        thresh_data = df[df['threshold'] == thresh]
        val1 = thresh_data.iloc[0][plot_metric]
        val2 = thresh_data.iloc[1][plot_metric]
        delta = ((val2 - val1) / val1) * 100
        
        x_pos = list(thresholds).index(thresh)
        y_pos = max(val1, val2) + val2 * 0.1 # Position slightly above the highest bar
          
        # # Get standard deviation for error bar
        # # Plot error bars for each bar separately
        # for i, (val, row) in enumerate(zip([val1, val2], [thresh_data.iloc[0], thresh_data.iloc[1]])):
        #     yerr = row['ratio_tissue_vs_black_attr_std']  # Get std for each row
        #     bar_x = x_pos - 0.2 + (i * 0.4)  # Position at -0.2 for first bar, +0.2 for second bar
        #     plt.errorbar(x=bar_x, y=val, yerr=yerr, fmt='none', color='gray', alpha=0.7)
        
        # Create bubble-style annotation
        plt.hlines(y=y_pos, xmin=x_pos-0.4, xmax=x_pos+0.4, color='gray', alpha=0.7)
        
        # Add percentage bubble
        plt.annotate(f'{delta:+.1f}%',
                    xy=(x_pos, y_pos),
                    xytext=(0, 0), 
                    textcoords='offset points',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        fc='white',
                        ec='gray',
                        alpha=1.0
                    ))
        plt.errorbar(x=x_pos, y=y_pos, yerr=0, fmt='none', color='gray', alpha=0.7) 
    
    # Customize plot
    plt.xlabel('Threshold min % tissue pixels per patch')
    # plt.ylabel(f'Average {plot_metric.replace("_", " ").title()}')
    plt.ylabel(f'Percentage black attribution relative to percentage noise')

    
    # Add noise percentage line plot
    noise_data = df.groupby('threshold')['perc_noise'].mean().reset_index()
    
    # Create twin axis for noise line
    ax2 = ax.twinx()
    
    # Use categorical x positions instead of numeric values
    x_positions = range(len(noise_data))
    noise_line = ax2.plot(
        x_positions,
        noise_data['perc_noise'],
        color='gray',
        alpha=0.5,
        linestyle='--',
        marker='o',
        label='Percentage Noise'
    )
    
    ax2.set_ylabel('Percentage Noise', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.grid(False)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, 
              title='Processing Method', loc='lower right', framealpha=1.0)
    
    # Remove original legend from first axis
    ax.get_legend().remove()
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(os.path.join(output_dir, f'integrated_gradients_{plot_metric}.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'integrated_gradients_{plot_metric}.svg'), 
                bbox_inches='tight')
    plt.close()


def main():
    base_config = {
        'output_dir': '/Users/merlijnvanbreugel/Documents/GitHub/Ditto/ai-playground/',
        'compartment': 'Total',
        'figsize': (10, 6),
        'colors': ['#00A087FF', '#91D1C2FF'],
        'font_scale': 1.2,
        'dpi': 300,
        'plot_metric': 'rel_black_attr'
    }
    
    # Generate integrated gradients plot
    ig_config = base_config.copy()
    ig_config['input_file'] = '/Users/merlijnvanbreugel/Documents/GitHub/Ditto/ai-playground/integrated_gradients_relative_black.csv'
    ig_config['plot_metric'] = 'rel_black_attr'
    plot_tissue_attribution(**ig_config)

if __name__ == "__main__":
    main()


