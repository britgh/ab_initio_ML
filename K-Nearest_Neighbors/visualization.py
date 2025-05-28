# 3d scatter plot testing + training result data visualization
# britgh last updated: 05.27.25

import matplotlib.pyplot as plt
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

    # altering visual elements
    colors = np.array(['r', 'g'])
    icons = np.array(['o', 's', '^'])

    # plot training data
    print(train_label)
    for row in range(len(train_set)):
        # print(train_label.iloc[row].get('class'))
        axes.scatter(train_set.iloc[row, 0], train_set.iloc[row, 1], train_set.iloc[row, 2],
                     c='gray', marker=icons[train_label.iloc[row].get('class')])

    # plot testing data
    for row in range(len(test_set)):
        axes.scatter(test_set.iloc[row, 0], test_set.iloc[row, 1], test_set.iloc[row, 2],
                     c=colors[test_label.iloc[row,2]], marker=icons[test_label.iloc[row,1]]) # markers reflect actual label (not pred)

    plt.show()