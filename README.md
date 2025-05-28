# ab_initio_ML
various machine learning models made from scratch

## *K-Nearest Neighbor*
### Files
+ [KNN.py](https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/KNN.py)
   main portion of KNN Machine Learning Model
+ [processing.py](https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/processing.py)
   functions for KNN (min-max normalization, euclidean distance, min/max, mode, accuracy)
+ [visualization.py](https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/visualization.py)
   3D scatter plot data visualization explaining classification and prediction results per run

### Process
+ **Dataset**: UCIML Iris Dataset
+ **Data Split**: random 20% for testing (% changeable lol)
+ **Normalization**: Min-Max normalization with training set and current testcase
+  **Testing**: Euclidean distances (ascending) from training set to testcase, selected mode class of K closest neighbors as prediction
+  **Evaluation**: Calculated accuracy (correct predictions / total test cases) and created visualization for results

### Sample Output
![image](https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/sample_visual.png)
![image](https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/sample_output.png)

## Multi-Linear Regression
in-progress :-)
