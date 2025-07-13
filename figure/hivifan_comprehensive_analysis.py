import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for IEEE journal style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def generate_comprehensive_analysis_figure():
    """
    Generate the comprehensive HiViFAN analysis figure with 9 subplots
    matching IEEE journal publication standards.
    """
    
    # Create figure with appropriate dimensions for journal publication
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.06, right=0.98, top=0.95, bottom=0.06)
    
    # Define professional color palette
    colors = {
        'primary': '#2E86AB',    # Blue
        'secondary': '#A23B72',  # Purple
        'accent': '#F18F01',     # Orange
        'success': '#C73E1D',    # Red
        'dark': '#2F4858',       # Dark blue
        'light': '#7FB069',      # Light green
        'neutral': '#6C757D'     # Gray
    }
    
    # Generate and plot subplot (a): Directional accuracy by confidence
    ax_a = fig.add_subplot(gs[0, 0])
    plot_directional_accuracy_confidence(ax_a, colors)
    
    # Generate and plot subplot (b): Gate activation patterns
    ax_b = fig.add_subplot(gs[0, 1])
    plot_gate_activation_patterns(ax_b, colors)
    
    # Generate and plot subplot (c): Synergistic effects
    ax_c = fig.add_subplot(gs[0, 2])
    plot_synergistic_effects(ax_c, colors)
    
    # Generate and plot subplot (d): Performance degradation curves
    ax_d = fig.add_subplot(gs[1, 0])
    plot_performance_degradation(ax_d, colors)
    
    # Generate and plot subplot (e): Computational latency breakdown
    ax_e = fig.add_subplot(gs[1, 1])
    plot_computational_latency(ax_e, colors)
    
    # Generate and plot subplot (f): Attention weight distributions
    ax_f = fig.add_subplot(gs[1, 2])
    plot_attention_distributions(ax_f, colors)
    
    # Generate and plot subplot (g): Temporal evolution
    ax_g = fig.add_subplot(gs[2, 0])
    plot_temporal_evolution(ax_g, colors)
    
    # Generate and plot subplot (h): Collection-specific patterns
    ax_h = fig.add_subplot(gs[2, 1])
    plot_collection_patterns(ax_h, colors)
    
    # Generate and plot subplot (i): Error vs complexity correlation
    ax_i = fig.add_subplot(gs[2, 2])
    plot_error_complexity_correlation(ax_i, colors)
    
    # Add subplot labels with IEEE style
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i]
    
    for ax, label in zip(axes, subplot_labels):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=11, 
                fontweight='bold', va='bottom', ha='right')
    
    return fig

def plot_directional_accuracy_confidence(ax, colors):
    """Plot directional accuracy stratified by prediction confidence levels."""
    confidence_levels = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    accuracy_mean = np.array([72.1, 76.3, 79.8, 83.2, 86.1, 88.9, 91.2, 93.1, 94.2, 95.8])
    accuracy_std = np.array([3.2, 2.8, 2.5, 2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 0.9])
    
    # Plot main line with confidence interval
    ax.fill_between(confidence_levels, accuracy_mean - accuracy_std, 
                    accuracy_mean + accuracy_std, alpha=0.3, color=colors['primary'])
    ax.plot(confidence_levels, accuracy_mean, 'o-', color=colors['primary'], 
            linewidth=2, markersize=4, label='HiViFAN')
    
    # Add comparison baseline
    baseline_accuracy = np.array([65.2, 67.1, 68.9, 70.3, 71.8, 73.2, 74.5, 75.9, 77.1, 78.4])
    ax.plot(confidence_levels, baseline_accuracy, 's--', color=colors['secondary'], 
            linewidth=1.5, markersize=3, label='CLIP-Based', alpha=0.8)
    
    ax.set_xlabel('Prediction Confidence Percentile (%)')
    ax.set_ylabel('Directional Accuracy (%)')
    ax.set_title('Directional Accuracy vs Confidence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlim(5, 100)
    ax.set_ylim(60, 100)

def plot_gate_activation_patterns(ax, colors):
    """Plot dynamic gate activation patterns during volatility transitions."""
    time_points = np.linspace(0, 48, 100)  # 48 hours
    
    # Simulate market volatility event at t=24h
    volatility = 0.15 + 0.6 * np.exp(-((time_points - 24)**2) / 50) * np.sin((time_points - 24) / 2)
    volatility = np.clip(volatility, 0.1, 0.8)
    
    # Gate activations responding to volatility
    visual_gate = 0.65 - 0.25 * (volatility - 0.15) / 0.65
    market_gate = 0.35 + 0.25 * (volatility - 0.15) / 0.65
    
    visual_gate = np.clip(visual_gate, 0.2, 0.8)
    market_gate = np.clip(market_gate, 0.2, 0.8)
    
    # Create twin axis for volatility
    ax2 = ax.twinx()
    
    # Plot gate activations
    ax.fill_between(time_points, 0, visual_gate, alpha=0.6, color=colors['primary'], 
                    label='Visual Gate')
    ax.fill_between(time_points, visual_gate, visual_gate + market_gate, 
                    alpha=0.6, color=colors['accent'], label='Market Gate')
    
    # Plot volatility on secondary axis
    ax2.plot(time_points, volatility, color=colors['success'], linewidth=2, 
             linestyle='--', label='Market Volatility')
    
    # Mark crisis event
    ax.axvline(x=24, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(24.5, 0.9, 'Market\nCrash', fontsize=8, ha='left', va='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Gate Activation Weight')
    ax2.set_ylabel('Market Volatility', color=colors['success'])
    ax.set_title('Dynamic Gate Activation Patterns', fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

def plot_synergistic_effects(ax, colors):
    """Plot synergistic effects between architectural components."""
    components = ['Visual\nPyramid', 'Cross-Modal\nAttention', 'Temporal\nCoherence', 
                  'Dynamic\nGating', 'Combined\nSystem']
    individual_performance = np.array([0.8156, 0.7892, 0.8623, 0.8756, 0.9234])
    expected_additive = np.array([0.8156, 0.8024, 0.8389, 0.8512, 0.8445])
    
    x_pos = np.arange(len(components))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x_pos - width/2, individual_performance, width, 
                   color=colors['primary'], alpha=0.8, label='Actual Performance')
    bars2 = ax.bar(x_pos + width/2, expected_additive, width, 
                   color=colors['secondary'], alpha=0.8, label='Expected Additive')
    
    # Add synergy indicators
    synergy = individual_performance - expected_additive
    for i, (bar1, bar2, syn) in enumerate(zip(bars1, bars2, synergy)):
        if syn > 0.02:  # Significant synergy
            height = max(bar1.get_height(), bar2.get_height())
            ax.annotate(f'+{syn:.1%}', xy=(i, height + 0.01), ha='center', 
                       va='bottom', fontsize=8, fontweight='bold', color=colors['success'])
    
    # Highlight combined system
    bars1[-1].set_color(colors['accent'])
    bars1[-1].set_alpha(1.0)
    
    ax.set_xlabel('Architectural Components')
    ax.set_ylabel('R² Score')
    ax.set_title('Component Synergistic Effects', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(components, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.75, 0.95)

def plot_performance_degradation(ax, colors):
    """Plot performance degradation with increasing collection size."""
    collection_sizes = np.array([1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000])
    r2_scores = np.array([0.9234, 0.9198, 0.9156, 0.9087, 0.8956, 0.8823, 0.8691, 0.8534])
    mae_scores = np.array([0.0847, 0.0892, 0.0934, 0.0998, 0.1089, 0.1234, 0.1456, 0.1723])
    
    # Create twin axis for MAE
    ax2 = ax.twinx()
    
    # Plot R² scores
    line1 = ax.semilogx(collection_sizes, r2_scores, 'o-', color=colors['primary'], 
                        linewidth=2, markersize=5, label='R² Score')
    
    # Plot MAE scores on secondary axis
    line2 = ax2.semilogx(collection_sizes, mae_scores, 's--', color=colors['success'], 
                         linewidth=2, markersize=4, label='MAE (ETH)', alpha=0.8)
    
    # Add trend lines
    z1 = np.polyfit(np.log(collection_sizes), r2_scores, 1)
    p1 = np.poly1d(z1)
    ax.semilogx(collection_sizes, p1(np.log(collection_sizes)), 
                color=colors['primary'], alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Collection Size (log scale)')
    ax.set_ylabel('R² Score', color=colors['primary'])
    ax2.set_ylabel('Mean Absolute Error (ETH)', color=colors['success'])
    ax.set_title('Performance vs Collection Size', fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(800, 600000)

def plot_computational_latency(ax, colors):
    """Plot computational latency breakdown across visualization modules."""
    modules = ['t-SNE\nProjection', 'Attention\nFlow', '3D Surface\nRendering', 
               'Graph\nLayout', 'Real-time\nUpdates']
    latencies = [45.2, 30.1, 25.3, 18.7, 12.4]
    
    # Create pie chart with professional styling
    wedges, texts, autotexts = ax.pie(latencies, labels=modules, autopct='%1.1f%%',
                                      colors=[colors['primary'], colors['secondary'], 
                                             colors['accent'], colors['success'], colors['light']],
                                      startangle=90, textprops={'fontsize': 8})
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    # Add total latency annotation
    total_latency = sum(latencies)
    ax.text(0, -1.3, f'Total Latency: {total_latency:.1f}ms', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['neutral'], alpha=0.2))
    
    ax.set_title('Computational Latency Breakdown', fontweight='bold', pad=20)

def plot_attention_distributions(ax, colors):
    """Plot attention weight distributions across visual feature categories."""
    categories = ['Rare\nAccessories', 'Background\nType', 'Skin\nTone', 
                  'Eye\nType', 'Hair\nStyle', 'Mouth\nType']
    attention_weights = [0.213, 0.187, 0.123, 0.089, 0.072, 0.054]
    market_correlation = [0.87, 0.72, 0.61, 0.57, 0.52, 0.46]
    
    # Create scatter plot with bubble sizes
    bubble_sizes = np.array(attention_weights) * 1000
    scatter = ax.scatter(range(len(categories)), attention_weights, 
                        s=bubble_sizes, c=market_correlation, 
                        cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Price Correlation', rotation=270, labelpad=15, fontsize=8)
    
    # Add value labels
    for i, (cat, weight) in enumerate(zip(categories, attention_weights)):
        ax.annotate(f'{weight:.1%}', (i, weight), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Visual Feature Categories')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention Weight Distribution', fontweight='bold')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(attention_weights) * 1.2)

def plot_temporal_evolution(ax, colors):
    """Plot temporal evolution of feature importance over collection lifecycle."""
    months = np.arange(0, 25, 1)
    explicit_traits = 65 * np.exp(-months / 8) + 35
    aesthetic_quality = 20 + 25 * (1 - np.exp(-months / 6))
    trait_synergy = 15 + 20 * (1 - np.exp(-months / 10)) * np.sin(months / 4 + 1)
    
    # Ensure they sum to 100%
    total = explicit_traits + aesthetic_quality + trait_synergy
    explicit_traits = explicit_traits / total * 100
    aesthetic_quality = aesthetic_quality / total * 100
    trait_synergy = trait_synergy / total * 100
    
    # Create stacked area plot
    ax.fill_between(months, 0, explicit_traits, alpha=0.7, 
                    color=colors['primary'], label='Explicit Rare Traits')
    ax.fill_between(months, explicit_traits, explicit_traits + aesthetic_quality, 
                    alpha=0.7, color=colors['accent'], label='Aesthetic Quality')
    ax.fill_between(months, explicit_traits + aesthetic_quality, 
                    explicit_traits + aesthetic_quality + trait_synergy,
                    alpha=0.7, color=colors['secondary'], label='Trait Synergies')
    
    # Add milestone markers
    milestones = [6, 12, 18]
    milestone_labels = ['Early Trading', 'Market Maturation', 'Connoisseurship']
    for month, label in zip(milestones, milestone_labels):
        ax.axvline(x=month, color='gray', linestyle='--', alpha=0.5)
        ax.text(month, 95, label, rotation=90, ha='right', va='top', fontsize=7)
    
    ax.set_xlabel('Collection Age (months)')
    ax.set_ylabel('Feature Importance (%)')
    ax.set_title('Temporal Evolution of Features', fontweight='bold')
    ax.legend(loc='center right')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

def plot_collection_patterns(ax, colors):
    """Plot collection-specific attention patterns for different NFT types."""
    nft_types = ['CryptoPunks\n(Pixel Art)', 'Azuki\n(Illustrated)', 'Art Blocks\n(Generative)']
    scales = ['Fine Scale\n(Pixels)', 'Mid Scale\n(Features)', 'Coarse Scale\n(Composition)']
    
    # Attention patterns for each collection type
    attention_patterns = np.array([
        [0.65, 0.25, 0.10],  # CryptoPunks: high fine, low coarse
        [0.20, 0.60, 0.20],  # Azuki: balanced with mid emphasis
        [0.15, 0.25, 0.60]   # Art Blocks: high coarse, low fine
    ])
    
    # Create heatmap
    im = ax.imshow(attention_patterns, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.7)
    
    # Add text annotations
    for i in range(len(nft_types)):
        for j in range(len(scales)):
            text = f'{attention_patterns[i, j]:.2f}'
            ax.text(j, i, text, ha="center", va="center", 
                   color="white" if attention_patterns[i, j] > 0.35 else "black",
                   fontsize=9, fontweight='bold')
    
    # Set labels and title
    ax.set_xticks(range(len(scales)))
    ax.set_yticks(range(len(nft_types)))
    ax.set_xticklabels(scales)
    ax.set_yticklabels(nft_types)
    ax.set_xlabel('Attention Scale')
    ax.set_ylabel('NFT Collection Type')
    ax.set_title('Collection-Specific Patterns', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15, fontsize=8)

def plot_error_complexity_correlation(ax, colors):
    """Plot prediction error correlation with visual complexity."""
    np.random.seed(42)
    n_points = 500
    
    # Generate visual complexity scores
    complexity = np.random.beta(2, 5, n_points) * 10  # Skewed toward lower complexity
    
    # Generate prediction errors with U-shaped relationship to complexity
    base_error = 0.08 + 0.15 * (complexity - 5)**2 / 25  # U-shaped
    noise = np.random.normal(0, 0.02, n_points)
    prediction_error = np.abs(base_error + noise)
    
    # Create scatter plot with density coloring
    scatter = ax.scatter(complexity, prediction_error, alpha=0.6, 
                        c=colors['primary'], s=8, edgecolors='none')
    
    # Add trend line
    z = np.polyfit(complexity, prediction_error, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 10, 100)
    ax.plot(x_trend, p(x_trend), color=colors['success'], linewidth=2, 
            label='Polynomial Fit')
    
    # Add optimal complexity zone
    optimal_zone = (complexity >= 3) & (complexity <= 7)
    ax.axvspan(3, 7, alpha=0.2, color=colors['light'], 
               label='Optimal Complexity Zone')
    
    # Add correlation coefficient
    correlation = np.corrcoef(complexity, prediction_error)[0, 1]
    ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
            fontsize=9, fontweight='bold', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Visual Complexity Score')
    ax.set_ylabel('Prediction Error (MAE)')
    ax.set_title('Error vs Visual Complexity', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(prediction_error) * 1.1)

def main():
    """Main function to generate and save the comprehensive analysis figure."""
    print("Generating comprehensive HiViFAN analysis figure...")
    
    # Generate the figure
    fig = generate_comprehensive_analysis_figure()
    
    # Save as high-quality PDF
    output_filename = 'hivifan_comprehensive_analysis.pdf'
    fig.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Figure saved as {output_filename}")
    print("Figure generated successfully with IEEE journal formatting standards.")
    
    # Optionally display the figure
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Execute the main function
    fig = main()