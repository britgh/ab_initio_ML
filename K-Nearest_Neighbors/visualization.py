# 3d scatter plot testing + training result data visualization
# britgh last updated: 05.27.25

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

def plot (train_set, test_set, train_label, all) :
    graph = plt.figure()
    axes = graph.add_subplot(111, projection='3d')  # 3D graph for 3 features

    # Set labels for the axes
    axes.set_title("UCIML Iris Dataset KNN Prediction Results")
    axes.set_xlabel('X: Sepal Length')
    axes.set_ylabel('Y: Petal Width')
    axes.set_zlabel('Z: Petal Length')
    plt.figtext(0.5, 0.825, "britgh: Each icon is a data sample. \nShape is real classification. Color is testcase prediction accuracy.",  ha="center", fontsize=9, color='darkblue')

    # encode labels (IS=0, IVi=1, IVe=2); allow downcasting (string -> int)
    test_label = all.replace({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}).infer_objects(copy=False)
    train_label = train_label.replace({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}).infer_objects(copy=False)

    # 1st legend: custom icons (real label, not predicted)
    icons = np.array(['o', 's', '^'])

    circle_icon = Line2D([0],[0], label='Iris-setosa', c='black', marker='o', markersize=10, linestyle='')
    square_icon = Line2D([0],[0], label='Iris-virginica', c='black', marker='s', markersize=10, linestyle='')
    triangle_icon = Line2D([0],[0], label='Iris-versicolor', c='black', marker='^', markersize=10, linestyle='')

    axes.add_artist(axes.legend(handles=[circle_icon, square_icon, triangle_icon], loc='lower left', bbox_to_anchor=(-0.2, 0.6), title='Class'))

    # 2nd legend: custom colors (results)
    colors = np.array(['r', 'g'])

    grey_guide = patches.Patch(color='gray', label='training data')
    green_guide = patches.Patch(color='green', label = 'true predict')
    red_guide = patches.Patch(color='red', label='false predict')

    plt.legend(handles=[grey_guide, green_guide, red_guide], loc='upper left', bbox_to_anchor=(-0.2, 0.6), title='Result')

    #plot training data
    for row in range(len(train_set)):
        axes.scatter(train_set.iloc[row, 0], train_set.iloc[row, 1], train_set.iloc[row, 2],
                     c='gray', marker=icons[train_label.iloc[row].get('class')], edgecolors='black')

    # plot testing data
    for row in range(len(test_set)):
        axes.scatter(test_set.iloc[row, 0], test_set.iloc[row, 1], test_set.iloc[row, 2],
                     c=colors[test_label.iloc[row,2]], marker=icons[test_label.iloc[row,1]])

    plt.show()