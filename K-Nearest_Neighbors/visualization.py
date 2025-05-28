# 3d scatter plot testing + training result data visualization
# britgh last updated: 05.27.25

import matplotlib.pyplot as plt
import numpy as np

def plot (train_set, test_set, train_label, all) :
    print(all)
    graph = plt.figure()
    axes = graph.add_subplot(111, projection='3d')  # 3D graph for 3 features

    # Set labels for the axes
    axes.set_xlabel('X Axis: Sepal Length')
    axes.set_ylabel('Y Axis: Petal Width')
    axes.set_zlabel('Z Axis: Petal Length')

    # encode classes, allow downcasting (string -> int)
    label_set = all.replace({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}).infer_objects(copy=False)
    train_label = train_label.replace({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}).infer_objects(copy=False)
    # print(test_label)

    # altering visual elements
    colors = np.array(['r', 'g'])
    icons = np.array(['o', 's', '^'])

    # plot training data
    for row in range(len(train_set)):
        axes.scatter(train_set.iloc[row, 0], train_set.iloc[row, 1], train_set.iloc[row, 2],
                     c='gray', marker=icons[train_label.astype(int)[row]])

    # plot testing data
    for row in range(len(test_set)):
        axes.scatter(test_set.iloc[row, 0], test_set.iloc[row, 1], test_set.iloc[row, 2],
                     c=colors[label_set[:,2].astype(int)[row]], marker=icons[label_set[:,1].astype(int)[row]]) # markers reflect actual label (not pred)

    plt.show()