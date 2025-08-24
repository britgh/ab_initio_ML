# Data Visualizations for Multi-Linear Regression Model Performance
# britgh last updated: 08.24.25

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Displays MPG predictions in comparison to actual MPG used (x-axis is sample index)
def performance (actual, predicted) :
    for row in range(predicted.shape[0]) :
        plt.scatter(row, actual[row], marker='o', color='orchid')
        plt.scatter(row, predicted.iloc[row], marker='*', color='navy')

    plt.title("UCIML MGP Dataset Multi-Linear Regression Model Results")
    plt.ylabel('Miles Per Gallon')
    plt.xlabel('Sample Index')

    labels = ['Predicted MPG', 'Actual MPG']
    colors = ['navy', 'orchid']
    lines = [Line2D([0], [0], color=c, linewidth=10) for c in colors]
    plt.legend(lines, labels)

    plt.show()

# Display change in MSE as gradient descent algo executes X iterations
# visualization.development(TEST_cost, turns) ---
# visualization.consistency() --- compare fold performance
