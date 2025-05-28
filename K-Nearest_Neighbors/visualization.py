# 3d scatter plot testing + training result data visualization
# britgh last updated: 05.27.25

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Lines2D
import numpy as np

def plot (train_set, test_set, train_label, all) :
    # print(all)
    graph = plt.figure()
    axes = graph.add_subplot(111, projection='3d')  # 3D graph for 3 features

    # Set labels for the axes
    axes.set_xlabel('X Axis: Sepal Length')
    axes.set_ylabel('Y Axis: Petal Width')
    axes.set_zlabel('Z Axis: Petal Length')

    # encode classes, allow downcasting (string -> int)
    test_label = all.replace({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}).infer_objects(copy=False)
    train_label = train_label.replace({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}).infer_objects(copy=False)
    # print(test_label)

    # custom legend
    colors = np.array(['r', 'g'])
    grey_guide = patches.Patch(color='gray', label='training data')
    green_guide = patches.Patch(color='green', label = 'true predict')
    red_guide = patches.Patch(color='red', label='false predict')

    # custom icons
    icons = np.array(['o', 's', '^'])
    circle_icon = Lines2D([0,0], label='Iris-setosa', c='black', marker='o')
    square_icon = Lines2D([0,0], label='Iris-virginica', c='black', marker='s')
    triangle_icon = Lines2D([0,0], label='Iris-versicolor', c='black', marker='^')

    plt.legend(handles=[grey_guide, green_guide, red_guide, circle_icon, square_icon, triangle_icon])

    # plot training data
    # print(train_label)
    for row in range(len(train_set)):
        # print(train_label.iloc[row].get('class'))
        # print(train_set)
        axes.scatter(train_set.iloc[row, 0], train_set.iloc[row, 1], train_set.iloc[row, 2],
                     c='gray', marker=icons[train_label.iloc[row].get('class')])

    # plot testing data
    for row in range(len(test_set)):
        # print(test_set)
        # print(test_label)
        axes.scatter(test_set.iloc[row, 0], test_set.iloc[row, 1], test_set.iloc[row, 2],
                     c=colors[test_label.iloc[row,2]], marker=icons[test_label.iloc[row,1]]) # markers reflect actual label (not pred)

    plt.show()