import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

def plot_tissue_attribution(
    input_file: str,
    output_dir: str,
    compartment: str = 'Total',
    figsize: tuple = (10, 6),
    colors: list = ['#00A087FF', '#91D1C2FF'],
    font_scale: float = 1.5,
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
    df = df[df['compartment'] == compartment]
    
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
    plt.ylabel(f'Ratio of tissue vs artifact pixel attribution')
    # plt.title('Tissue Attribution Analysis by Threshold and Data Type')
    plt.legend(title='Processing Method', loc='lower right', framealpha=1.0)
    plt.tight_layout()
    
    # Save plots
    plt.savefig(os.path.join(output_dir, f'integrated_gradients_{plot_metric}.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'integrated_gradients_{plot_metric}.svg'), 
                bbox_inches='tight')
    plt.close()

def plot_gradcam_attribution(
    input_file: str,
    output_dir: str,
    compartment: str = 'Total',
    figsize: tuple = (10, 6),
    colors: list = ['#00A087FF', '#91D1C2FF'],
    font_scale: float = 1.2,
    dpi: int = 300,
    plot_metric: str = 'avg_ratio_tissue_vs_black'
) -> None:
    """
    Plot GradCAM attribution analysis.
    
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
    df = df[df['compartment'] == compartment]
    
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
        y=plot_metric,  # Changed from mean_black_attribution
        hue='data_type',
        palette=colors,
        errorbar=None,
        ax=ax
    )
    
    # Calculate and add delta indicators
    thresholds = df['threshold'].unique()
    for thresh in thresholds:
        thresh_data = df[df['threshold'] == thresh]
        # print(thresh_data)
        val1 = thresh_data.iloc[0][plot_metric]
        val2 = thresh_data.iloc[1][plot_metric]
        delta = ((val2 - val1) / val1) * 100
        
        x_pos = list(thresholds).index(thresh)
        y_pos = max(val1, val2) + val2 * 0.1
        
        plt.hlines(y=y_pos, xmin=x_pos-0.4, xmax=x_pos+0.4, color='gray', alpha=0.7)
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
    
    # Customize plot
    plt.xlabel('Threshold min % tissue pixels per patch')
    plt.ylabel(f'Ratio of tissue vs artifact pixel attribution')
    plt.legend(title='Processing Method', loc='lower right', framealpha=1.0)
    plt.tight_layout()

    print(f'Saving plot to {os.path.join(output_dir, f"gradcam_{plot_metric}.png")}')
    # Save plots
    plt.savefig(os.path.join(output_dir, f'gradcam_{plot_metric}.png'),  # Changed filename
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'gradcam_{plot_metric}.svg'),  # Changed filename
                bbox_inches='tight')
    plt.close()

def perform_mixed_effects_analysis(
    input_file: str,
    mapping_file: str = '/workspace/ImageRecognition/1_data/mapping.csv',
    compartment: str = 'Total',
    plot_metric: str = 'perc_black_attr_total_attribution_median'
) -> dict:
    """
    Perform mixed-effects analysis comparing random vs black methods.
    
    Args:
        input_file: Path to the integrated gradients CSV file
        mapping_file: Path to the mapping CSV file
        compartment: Compartment to analyze
        plot_metric: Metric to analyze
    
    Returns:
        Dictionary containing statistical test results
    """
    # Read and prepare data
    df = pd.read_csv(input_file)
    df = df[df['compartment'] == compartment]
    
    # Extract tiff filename and prepare identifiers
    df['tiff'] = df['filename'].apply(lambda x: Path(x).stem)
    
    # Read and prepare mapping data
    mapping = pd.read_csv(mapping_file, sep=';')
    df = df.merge(mapping[['tiff', 'patient', 'airway']], on='tiff', how='left')
    
    # Create binary indicator for random imputation
    df['is_random'] = (df['data_type'] == 'random_image').astype(int)
    
    # Prepare data for mixed model
    model_data = df[[plot_metric, 'is_random', 'patient', 'tiff', 'airway', 'threshold']]
    
    results = {}
    
    # Perform analysis for each threshold
    for thresh in df['threshold'].unique():
        thresh_data = model_data[model_data['threshold'] == thresh].copy()
        
        # Create nested random effects grouping
        thresh_data['group_id'] = (thresh_data['patient'] + '_' + 
                                 thresh_data['airway'] + '_' + 
                                 thresh_data['tiff'])
        
        try:
            # Fit mixed-effects model
            model = MixedLM(
                endog=thresh_data[plot_metric],
                exog=sm.add_constant(thresh_data['is_random']),
                groups=thresh_data['group_id']
            )
            
            model_fit = model.fit()
            
            # Extract results
            results[thresh] = {
                'coefficient': model_fit.params['is_random'],
                'std_err': model_fit.bse['is_random'],
                'pvalue': model_fit.pvalues['is_random'],
                'conf_int': model_fit.conf_int().loc['is_random'].tolist(),
                'n_samples': len(thresh_data),
                'n_groups': len(thresh_data['group_id'].unique()),
                'n_patients': len(thresh_data['patient'].unique()),
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
            
        except Exception as e:
            print(f"Error fitting model for threshold {thresh}: {str(e)}")
            results[thresh] = None
            
    return results

def print_mixed_effects_results(results: dict) -> None:
    """Print formatted mixed-effects model results."""
    print("\nMixed-Effects Model Results")
    print("===========================")
    
    for thresh, data in results.items():
        if data is None:
            print(f"\nThreshold {thresh}: Model fitting failed")
            continue
            
        print(f"\nThreshold: {thresh}")
        print("-------------------")
        print(f"Samples: {data['n_samples']}")
        print(f"Unique groups: {data['n_groups']}")
        print(f"Unique patients: {data['n_patients']}")
        print("\nRandom imputation effect:")
        print(f"Coefficient: {data['coefficient']:.3f}")
        print(f"Standard Error: {data['std_err']:.3f}")
        print(f"P-value: {data['pvalue']:.3e}")
        print(f"95% CI: [{data['conf_int'][0]:.3f}, {data['conf_int'][1]:.3f}]")
        print(f"AIC: {data['aic']:.1f}")
        print(f"BIC: {data['bic']:.1f}")

def main():
    base_config = {
        'output_dir': '/workspace/ImageRecognition/5_results/learning_evaluation/completed_grid_runs_20241118_123457_threshold_selection_new/information_plots',
        'compartment': 'Total',
        'figsize': (10, 6),
        'colors': ['#00A087FF', '#91D1C2FF'],
        'font_scale': 1.2,
        'dpi': 300,
        'plot_metric': 'avg_ratio_tissue_vs_black'
    }
    
    # Generate integrated gradients plot
    ig_config = base_config.copy()
    ig_config['input_file'] = str(Path(ig_config['output_dir']).parent / 'integrated_gradients_analysis_aggregated.csv')
    ig_config['plot_metric'] = 'perc_black_attr_total_attribution_median'
    plot_tissue_attribution(**ig_config)
    
    # Generate GradCAM plot
    gradcam_config = base_config.copy()
    gradcam_config['input_file'] = str(Path(gradcam_config['output_dir']).parent / 'gradcam_analysis_aggregated.csv')
    gradcam_config['plot_metric'] = 'perc_black_attr_total_attribution_median'
    plot_gradcam_attribution(**gradcam_config)
    
    # Add after the integrated gradients plot generation:
    results = perform_mixed_effects_analysis(
        input_file=ig_config['input_file'],
        plot_metric=ig_config['plot_metric']
    )
    print_mixed_effects_results(results)

if __name__ == "__main__":
    main()