import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Set up the plot style
plt.style.use('ggplot')

# Configure Matplotlib to use a serif font (e.g., Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Data
configs = ['W3A3', 'W3A6', 'W3A8', 'W4A4', 'W4A6', 'W4A8', 
           'W5A5', 'W5A6', 'W5A8', 'W6A6', 'W6A7', 'W6A8']
top1_acc = [63.24, 64.92, 64.86, 67.83, 68.26, 68.22, 
            69.10, 69.27, 69.21, 69.58, 69.57, 69.62]

# Original mixed-precision results (baselines)
baselines = {
    'W3A3': 64.92,
    'W4A4': 67.57,
    'W6A6': 70.23
}

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Set the width of each bar and positions of the bars
width = 0.6  # Increased width for better visibility
x = np.arange(len(configs))

# Color scheme for different weight bitwidths (colorblind-friendly)
colors = ['#377eb8', '#4daf4a', '#e41a1c', '#984ea3']  # Blue, Green, Red, Purple

# Initialize lists to store legend handles and labels
legend_labels = []
custom_handles = []

# Create the bars with different colors for each weight bitwidth
for i, (start, end) in enumerate([(0, 3), (3, 6), (6, 9), (9, 12)]):
    # Plot Top-1 Accuracy bars
    rects = ax.bar(x[start:end], top1_acc[start:end], width,
                  color=colors[i], alpha=0.8)
    
    # Create custom legend handles
    if i < len(colors):
        custom_handles.append(Patch(facecolor=colors[i], edgecolor='none'))
        legend_labels.append(f'W{i+3} Top-1')
    
    # Add baseline for the first bar in each group if available
    config = configs[start]
    if config in baselines:
        baseline = baselines[config]
        # Plot baseline as a separate bar (gray color with hatch pattern)
        baseline_rect = ax.bar(x[start] + width * 0.15, baseline, width * 0.7,
                               color='gray', alpha=0.5, hatch='//', edgecolor='gray')
        # Annotate baseline value
        # Since labels are to be removed, we skip adding text annotations
        # If baseline annotations are still desired, uncomment the following lines:
        # ax.text(x[start] + width * 0.15, baseline + 0.5, f'{baseline:.2f}',
        #         ha='center', va='bottom', fontsize=10, rotation=90)

# Customize the axis
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Top-1 Accuracy Comparison for Different Configurations\nwith Original Mixed-Precision Baselines', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=45, ha='right')

# Remove the legend
# If you still want to include the legend, comment out the following two lines
# and uncomment the legend-related code.
# ax.legend(custom_handles, legend_labels, ncol=4, loc='upper center',
#           bbox_to_anchor=(0.5, 1.15), frameon=False)

# Add vertical lines to separate weight bitwidth groups
for i in range(1, 4):
    ax.axvline(x=i*3 - 0.5, color='gray', linestyle='--', alpha=0.7)

# Improve layout to accommodate the legend (if present)
fig.tight_layout(rect=[0, 0, 1, 1])  # Adjusted rect since legend is removed

# Save the figure with high resolution
plt.savefig('top1_accuracy_comparison_bar_chart_with_baselines.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('top1_accuracy_comparison_bar_chart_with_baselines.png',
            bbox_inches='tight', dpi=300)  # Also save as PNG if needed
plt.close()
