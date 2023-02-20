# Machine-Learning
In this repository, you can find my work related to Machine Learning topics like Naive Bayesian classification on 20 Newsgroups dataset, KNeighbors classification on MNIST dataset etc.. 

KNN_MNIST.ipynb consists Exercise problems (1, 2) for Classification Topic (Chapter-3) in "Hands-on Machine Learning with Scikit-Learn, Keras and Tensor-Flow"
Note: Beginning of the code till exercises is just a practice along with the book and only difference is the usage of different digit image (4 in this case)

Steps followed for Exercise-1: 

I followed the following steps to tune hyperparameters for a k-nearest neighbors (KNN) classifier and evaluate its performance on a test set:

•	Import the necessary libraries: import the GridSearchCV and accuracy_score functions from scikit-learn's model_selection and metrics modules, respectively.

•	Define the hyperparameter grid: Dictionary param_grid specifies the hyperparameters to be tuned (n_neighbors and weights) and the range of values to search over for each hyperparameter.

•	Initialize the grid search object: GridSearchCV object called grid_search that takes as input the KNN classifier object knn_clf, the hyperparameter grid param_grid, and other parameters such as the number of cross-validation folds (cv=5) and the number of CPU cores to use for parallelization (n_jobs=-1).

•	Fit the grid search object to the training data: The code fits the grid_search object to the training data (X_train and y_train) using the fit() method.

•	Best hyperparameters: Best hyperparameters were found during the grid search using the best_params_ attribute of the grid_search object.

•	Evaluate the model on the test set: Predictions on the test set (X_test) using the model found during the grid search (grid_search.predict(X_test)), and calculate the accuracy of the predictions using the accuracy_score() function. The resulting accuracy is 97.3%

 

Overall, this process involves tuning hyperparameters for a KNN classifier using a grid search with cross-validation and evaluating the resulting model on the test set. By selecting the hyperparameters that yield the best cross-validation performance, this approach aims to maximize the generalization performance of the model on unseen data.

Steps followed for Exercise-2: 

I followed the following process to complete this task:
•	Definition of a function shift_image that shifts a given MNIST image in any of the four directions (left, right, up, or down) by one pixel.
 

•	Next, an image is selected from the training set using its index (X_train[1003]), and two shifted copies of the image are created using the shift_image function defined earlier. One of the shifted images is shifted down (shift_image(image, 'down')) and the other is shifted left (shift_image(image, 'left')).

•	Next, a figure with three subplots is created using the plt.subplots() function. The first subplot displays the original image, while the second and third subplots display the shifted images.
 

•	This visualization allows us to see the effect of the data augmentation process on a single image and helps to illustrate how the augmented data can help to improve the robustness and generalization ability of the classifier.

•	Perform data augmentation on MNIST training set by creating four shifted copies of each image in the set and appending them to the original training set. This can help to improve the accuracy and generalization ability of the model by making it more robust to variations and distortions in the input data.

•	Shuffling of the augmented training set using np.random.permutation. Shuffling the data is a standard technique in machine learning to ensure that the model does not overfit to any particular ordering of the training set, and to ensure that it generalizes well to new data.

•	Training of a K-nearest neighbors (KNN) classifier on the augmented training set using the best hyperparameters found by a grid search with cross-validation.

•	Evaluation of the trained KNN classifier on the test set using the accuracy_score, recall_score, precision_score, and f1_score functions from sklearn.metrics module.

•	An interesting fact is that this model results in producing 97.9% for all the metrics (accuracy score, precision score, recall score and f1 score) 
 

Overall, this process involves data augmentation to expand the training set, training of a KNN classifier using the augmented data, and evaluation of the classifier. 
