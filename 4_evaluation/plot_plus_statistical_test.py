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
    plot_metric: str = 'ratio_tissue_vs_black_attr',
    results: dict = None
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
        results: Dictionary containing statistical test results
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
    
    # Add noise pixels line plot on same axis
    noise_data = df[df['data_type'] == 'Random Imputation'].groupby('threshold')['noise_pixels_perc_mean'].mean()
    ax.plot(range(len(df['threshold'].unique())), noise_data.values, color='gray', alpha=0.5, 
            linestyle='--', marker='o', markersize=4, label='Noise Pixels %')
    
    # Update legend to include noise line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Processing Method', 
             loc='lower right', framealpha=1.0)
    
    # Calculate and add delta indicators
    thresholds = df['threshold'].unique()
    for thresh in thresholds:
        thresh_data = df[df['threshold'] == thresh]
        val1 = thresh_data.iloc[0][plot_metric]
        val2 = thresh_data.iloc[1][plot_metric]
        delta = ((val2 - val1) / val1) * 100
        
        x_pos = list(thresholds).index(thresh)
        y_pos = max(val1, val2) + val2 * 0.1
        
        # Add significance stars if results are provided
        significance_str = ''
        if results and thresh in results and results[thresh]:
            p_value = results[thresh]['pvalue']
            if p_value < 0.001:
                significance_str = '***'
            elif p_value < 0.01:
                significance_str = '**'
            elif p_value < 0.05:
                significance_str = '*'
        
        plt.hlines(y=y_pos, xmin=x_pos-0.4, xmax=x_pos+0.4, color='gray', alpha=0.7)
        
        # Add percentage bubble with significance stars
        plt.annotate(f'{delta:+.1f}%{significance_str}',
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
    if plot_metric == 'perc_black_attr_total_attribution_median':
        plt.ylabel('Learning attribution to masked pixels')
    elif plot_metric == 'ratio_tissue_vs_black_attr_median':
        plt.ylabel('Ratio of tissue vs. masked attribution')
    else:
        plt.ylabel(f'{plot_metric.replace("_", " ").title()}')
    plt.tight_layout()
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    
    # Save plots
    plt.savefig(os.path.join(output_dir, f'integrated_gradients_{plot_metric}.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'integrated_gradients_{plot_metric}.svg'), 
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'integrated_gradients_{plot_metric}.pdf'), 
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
    plot_metric: str = 'avg_ratio_tissue_vs_black',
    results: dict = None
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
        results: Dictionary containing statistical test results
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
    
    # Read noise pixels data from integrated gradients file
    ig_file = input_file.replace('gradcam', 'integrated_gradients')
    ig_df = pd.read_csv(ig_file)
    ig_df = ig_df[ig_df['compartment'] == compartment]
    noise_data = ig_df[ig_df['data_type'] == 'random_image'].groupby('threshold')['noise_pixels_perc_mean'].mean()
    
    # Add noise pixels line plot on same axis
    ax.plot(range(len(df['threshold'].unique())), noise_data.values, color='gray', alpha=0.5, 
            linestyle='--', marker='o', markersize=4, label='Noise Pixels %')
    
    # Update legend to include noise line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Processing Method', 
             loc='lower right', framealpha=1.0)
    
    # Calculate and add delta indicators
    thresholds = df['threshold'].unique()
    for thresh in thresholds:
        thresh_data = df[df['threshold'] == thresh]
        val1 = thresh_data.iloc[0][plot_metric]
        val2 = thresh_data.iloc[1][plot_metric]
        delta = ((val2 - val1) / val1) * 100
        
        x_pos = list(thresholds).index(thresh)
        y_pos = max(val1, val2) + val2 * 0.1
        
        # Add significance stars if results are provided
        significance_str = ''
        if results and thresh in results and results[thresh]:
            p_value = results[thresh]['pvalue']
            if p_value < 0.001:
                significance_str = '***'
            elif p_value < 0.01:
                significance_str = '**'
            elif p_value < 0.05:
                significance_str = '*'
        
        plt.hlines(y=y_pos, xmin=x_pos-0.4, xmax=x_pos+0.4, color='gray', alpha=0.7)
        plt.annotate(f'{delta:+.1f}%{significance_str}',
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
    if plot_metric == 'perc_black_attr_total_attribution_median':
        plt.ylabel('Learning attribution to masked pixels')
    elif plot_metric == 'ratio_tissue_vs_black_attr_median':
        plt.ylabel('Ratio of tissue vs. black attribution')
    else:
        plt.ylabel(f'{plot_metric.replace("_", " ").title()}')
    plt.tight_layout()

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))

    print(f'Saving plot to {os.path.join(output_dir, f"gradcam_{plot_metric}.png")}')
    # Save plots
    plt.savefig(os.path.join(output_dir, f'gradcam_{plot_metric}.png'),
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'gradcam_{plot_metric}.svg'),
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'gradcam_{plot_metric}.pdf'),
                bbox_inches='tight')
    plt.close()

def perform_mixed_effects_analysis(
    input_file: str,
    mapping_file: str = './1_data/mapping.csv',
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
    print(f"\nStarting mixed-effects analysis:")
    print(f"Input file: {input_file}")
    print(f"Mapping file: {mapping_file}")
    print(f"Compartment: {compartment}")
    print(f"Plot metric: {plot_metric}")
    
    # Read and prepare data
    print("\nReading and preparing data...")
    df = pd.read_csv(input_file)
    print(f"Initial dataframe shape: {df.shape}")
    
    # df = df[df['compartment'] == compartment]
    # print(f"After compartment filtering shape: {df.shape}")
    
    # Extract tiff filename and prepare identifiers
    df['tiff'] = df['filename'].apply(lambda x: Path(x).stem.split('__Area_')[0])
    print(f"Unique tiff files: {len(df['tiff'].unique())}")
    
    # Read and prepare mapping data
    print("\nReading mapping data...")
    mapping = pd.read_csv(mapping_file, sep=';')
    print(f"Mapping data shape: {mapping.shape}")
    
    # Check for unmatched tiffs before merge
    df_tiffs = set(df['tiff'].unique())
    mapping_tiffs = set(mapping['tiff'].unique())
    unmatched_tiffs = df_tiffs - mapping_tiffs
    
    print(f"Number of unmatched tiffs: {len(unmatched_tiffs)}")
    if unmatched_tiffs:
        print("\nWARNING: Found tiffs in df that don't exist in mapping file:")
        print(f"Number of unmatched tiffs: {len(unmatched_tiffs)}")
        print("First 5 unmatched tiffs:", list(unmatched_tiffs)[:5])
    
    # Perform merge and check for null values
    df = df.merge(mapping[['tiff', 'patient']], on='tiff', how='left')
    null_counts = df['tiff'].isnull().sum()
    if null_counts > 0:
        print(f"\nWARNING: Found {null_counts} rows with null values after merge")
        
    print(f"After merging with mapping data shape: {df.shape}")
    # print(f"Number of unique patients: {len(df['patient'].unique())}")
    
    # Print first 10 rows of the merged dataframe
    print("\nFirst 10 rows of merged dataframe:")
    pd.set_option('display.max_columns', None)  # Show all columns
    print(df.head(10))
    pd.reset_option('display.max_columns')  # Reset to default
    # Create binary indicator for random imputation
    df['is_random'] = (df['data_type'] == 'random_image').astype(int)
    print(f"\nDistribution of random vs black images:")
    print(df['is_random'].value_counts())
    
    # Prepare data for mixed model
    model_data = df[[plot_metric, 'is_random', 'patient', 'tiff', 'threshold']]
    
    results = {}
    print("\nStarting threshold-specific analyses...")
    
    # Perform analysis for each threshold
    for thresh in df['threshold'].unique():
        print(f"\nAnalyzing threshold {thresh}...")
        thresh_data = model_data[model_data['threshold'] == thresh].copy()
        print(f"Threshold data shape: {thresh_data.shape}")
        
        # Create nested random effects grouping by converting to strings
        thresh_data['group_id'] = thresh_data['patient'].astype(str) + '_' + thresh_data['tiff'].astype(str)
        print(f"Number of unique groups: {len(thresh_data['group_id'].unique())}")
        
        try:
            print("Fitting mixed-effects model...")
            # Fit mixed-effects model
            model = MixedLM(
                endog=thresh_data[plot_metric],
                exog=sm.add_constant(thresh_data['is_random']),
                groups=thresh_data['group_id']
            )
            
            model_fit = model.fit()
            print("Model fitting successful")
            
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
                'bic': model_fit.bic,
                'model_summary': model_fit.summary()
            }
            
        except Exception as e:
            print(f"Error fitting model for threshold {thresh}")
            print(f"Error details: {str(e)}")
            print("Data summary for debugging:")
            print(thresh_data.describe())
            results[thresh] = None
            
    print("\nAnalysis complete!")
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
        print("=" * 50)
        print(f"Data Summary:")
        print(f"  Samples: {data['n_samples']}")
        print(f"  Unique groups: {data['n_groups']}")
        print(f"  Unique patients: {data['n_patients']}")
        
        print("\nModel Results:")
        print("  Random imputation effect:")
        print(f"    Coefficient: {data['coefficient']:.3f}")
        print(f"    Standard Error: {data['std_err']:.3f}")
        print(f"    P-value: {data['pvalue']:.3e}")
        print(f"    95% CI: [{data['conf_int'][0]:.3f}, {data['conf_int'][1]:.3f}]")
        
        print("\nModel Fit Statistics:")
        print(f"  AIC: {data['aic']:.1f}")
        print(f"  BIC: {data['bic']:.1f}")
        
        print("\nDetailed Model Summary:")
        print(data['model_summary'])
        print("\n" + "=" * 50)

def main():
    base_config = {
        'output_dir': '/Users/merlijnvanbreugel/Documents/GitHub/ImageRecognition/5_results/learning_evaluation/completed_grid_runs_20241118_123457_threshold_selection_new/information_plots',
        'compartment': 'Total',
        'figsize': (10, 6),
        'colors': ['#00A087FF', '#91D1C2FF'],
        'font_scale': 1.2,
        'dpi': 300,
        'plot_metric': 'perc_black_attr_total_attribution_median'
    }
    
    # Integrated Gradients analysis and plotting
    ig_config = base_config.copy()
    ig_config['input_file'] = '/Users/merlijnvanbreugel/Documents/GitHub/ImageRecognition/5_results/learning_evaluation/completed_grid_runs_20241118_123457_threshold_selection_new/integrated_gradients_analysis_all_results.csv'
    ig_stats_results = perform_mixed_effects_analysis(
        input_file=ig_config['input_file'],
        plot_metric='perc_black_attr_total_attribution'
    )
    print("Mixed effects analysis results of integrated gradients:")
    print_mixed_effects_results(ig_stats_results)

    ig_config['input_file'] = str(Path(ig_config['output_dir']).parent / 'integrated_gradients_analysis_aggregated.csv')
    ig_config['plot_metric'] = 'perc_black_attr_total_attribution_median'
    plot_tissue_attribution(**ig_config, results=ig_stats_results)
    
    # GradCAM analysis and plotting
    gradcam_config = base_config.copy()
    gradcam_config['input_file'] = str(Path(gradcam_config['output_dir']).parent / 'gradcam_analysis_all_results.csv')
    gradcam_stats_results = perform_mixed_effects_analysis(
        input_file=gradcam_config['input_file'],
        plot_metric='perc_black_attr_total_attribution'
    )
    print("Mixed effects analysis results of GradCAM:")
    print_mixed_effects_results(gradcam_stats_results)

    gradcam_config['input_file'] = str(Path(gradcam_config['output_dir']).parent / 'gradcam_analysis_aggregated.csv')
    gradcam_config['plot_metric'] = 'perc_black_attr_total_attribution_median'
    plot_gradcam_attribution(**gradcam_config, results=gradcam_stats_results)

if __name__ == "__main__":
    main()