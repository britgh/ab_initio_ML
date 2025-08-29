# ab_initio_ML
building various machine learning models from 'scratch' (NumPy + pandas)

---
## *K-Nearest Neighbor (KNN)*
### Files
+ [KNN.py](https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/KNN.py)
   main portion of KNN Machine Learning Model
+ [processing.py](https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/processing.py)
   general functions for KNN (min-max normalization, euclidean distance, min/max, mode, accuracy)
+ [visualization.py](https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/visualization.py)
   3D scatter plot data visualization explaining classification and prediction results per run

### Process
+ **Dataset**: UCIML Iris Dataset
+ **Data Split**: 80% training, 20% testing
+ **Normalization**: Min-Max normalization with training set and current testcase
+  **Testing**: Euclidean distances (ascending) from training set to testcase, selected mode class of K closest neighbors as prediction
+  **Evaluation**: Calculated accuracy (correct predictions / total test cases) and created visualization for results

### Sample
<img src="https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/sample_visual.png" width="350" />
<br>
<img src="https://github.com/britgh/ab_initio_ML/blob/main/K-Nearest_Neighbors/sample_output.png" width="350" />

---
## *Multi-Linear Regression (MLR)*
### Files
+ [linear_regression.py](https://github.com/britgh/ab_initio_ML/blob/main/Multi-Linear_Regression/linear_regression.py)
   main portion of Multi-Linear Regression model's operations, KFold
+ [calculations.py](https://github.com/britgh/ab_initio_ML/blob/main/Multi-Linear_Regression/calculations.py)
   mathematical functions (prediction, MSE, partial derivatives, gradient descent, R2 score)
+ [correlation.py](https://github.com/britgh/ab_initio_ML/blob/main/Multi-Linear_Regression/correlation.py)
    Tidied data and used matplotlib (DataViz lib) + PCC to preprocess data (determine removable features)
+ [visualization.py](https://github.com/britgh/ab_initio_ML/blob/main/Multi-Linear_Regression/visualization.py)
   2D scatterplot: X=sample_index, Y=score (true + predicted differentiated via color)

### Process
+ **Dataset**: UCIML MPG Dataset
+ **Data Split**: K-Fold Cross-Validation (K=10)
+ **Normalization**: Z-score normalization w/ mean + sd from training data used to normalize testing data
+ **Training**: Default weights vector + bias to 0, measured MSE, calculated w + b partial derivatives, updated w + b, iterated for X rounds (returned costs and final w + b)
+ **Testing**: Predicted label using X_testing data w/ weights and bias passed in from training
+  **Evaluation**: Calculated R2 score (= 1/(RSS-TSS)) to determine accuracy and created data visualization plotting predicted v. true scores

### Sample
<img src="https://github.com/britgh/ab_initio_ML/blob/main/Multi-Linear_Regression/sample_fold_result.png" width="350" />
<br>
<img src="https://github.com/britgh/ab_initio_ML/blob/main/Multi-Linear_Regression/sample_fold_result.png" width="350" />

---
## *Neural Network*
in-progress :-)
