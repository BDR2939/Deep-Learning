r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. False. The test set allows us to estimate our generalization error. The in-sample error is the error of the model on the training set.
2. False. The usefulness of a train-test split depends on a number of factors, including the size of the dataset, the distribution of the data, and the specific machine learning algorithm being used. For example, if the dataset is small, then it is important to make sure that the train and test sets are both representative of the overall dataset. If the data is not evenly distributed, then it is important to make sure that the train and test sets have similar distributions. And if the machine learning algorithm is sensitive to outliers, then it is important to make sure that the train and test sets do not contain any outliers.
3. True. The purpose of cross-validation is to evaluate the performance of a machine learning model on data that it has not seen before. The test set is used to evaluate the final model after it has been tuned. If the test set is used during cross-validation, then the model will be tuned to perform well on the test set, which will not be representative of the data that it will see in the real world and it may lead to overfitting.
4. True. This is because the validation set is not used to train the model, so it provides an unbiased estimate of the model's performance on new data.
"""

part1_q2 = r"""
Yes, his approach is justified. The purpose of regularization is to prevent a model from overfitting the training data. Regularization helps to prevent overfitting by penalizing the model for having large weights. This forces the model to find a simpler solution that is less likely to overfit the training data. However, he shouldn't have used the test set to tune the hyperparameters of the model, for that he should have used cross-validation. The test set should only be used to evaluate the final model after it has been tuned. 
"""

# ==============
# Part 2 answers

part2_q1 = r"""
Increasing k can improve generalization up to a point. Beyond that point, increasing k will decrease generalization. The optimal value of k depends on the dataset. It is best to start with a small value of k and then increase it until the model starts to overfit the training data. Once the model starts to overfit, it is best to decrease k slightly.
The extremal values of k are 1 and the number of data points in the training set. A value of k = 1 will lead to a model that is very specific and will likely overfit the training data. A value of k = the number of data points in the training set will lead to a model that is very general and will likely underfit the training data.
"""

part2_q2 = r"""
1. It is less likely to overfit the training data, and it provides a more accurate estimate of the model's performance on unseen data. When we train a model on the entire train set and select the best model based on its performance on the train set, we run the risk of selecting a model that fits the noise in the data rather than the underlying patterns. This can result in a model that performs well on the train set but poorly on unseen data.
2. It provides a more accurate estimate of the model's performance on unseen data. Selecting the best model with respect to test-set accuracy can be biased, especially if the same test set is used repeatedly to evaluate the performance of different models. This is because the test set becomes part of the training process, and the performance estimates obtained on the test set can be overly optimistic.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
The selection of the margin is somewhat arbitrary for the hinge loss because there is no clear mathematical or statistical justification for choosing a particular value. The margin is a hyperparameter that needs to be set based on empirical performance.
"""

part3_q2 = r"""
1. The weights assigned to each feature can indicate the importance of that feature in separating the different classes. For example, if the model assigns a high weight to the pixel values in the center of the image, this might indicate that the model is relying heavily on the presence of a line in the center of the digit to classify it correctly. Therefore, if some outlier sample is not following the common pattern of its class it is likely to not get classified correctly. Such an example is the wrong classification of 5 as 6 in the first row of the visualization.
2. In KNN, the classification decision is based on the majority class of the k nearest neighbors of a given test sample in the feature space. There are no explicit feature weights or coefficients involved in the KNN algorithm, so we cannot directly interpret the importance of features from the model itself.
"""

part3_q3 = r"""
1. The chosen learning rate may be not optimal, however it is in the range of good learning rates since the loss graph exhibits a smooth and steady decrease over the training epochs. If the learning rate is too high, the model may miss the optimal point of the loss function and fail to converge. On the other hand, if the learning rate is too low, the model may converge very slowly to a minimum, and it may take a long time to achieve a satisfactory level of performance.
2. Slightly overfitted to the training set. When a model is slightly overfitted to the training set, we expect to see that the training set accuracy is higher than the test set accuracy, indicating that the model is fitting the training set well but not generalizing well to unseen data. Additionally, we expect to see that the training set accuracy is increasing steadily. At the same time, the test set accuracy would also increase but at a slower rate and eventually plateau or even start decreasing. The accuracy graph reflects exactly these two observations.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
In an ideal residual plot, the residuals should be randomly scattered around the horizontal line at y=0. We can see that compared to the plot for the top-5 features, in the final plot after CV the error is closer to the y=0 axis. This is reflected by the distance of the margin from y=0. 
"""

part4_q2 = r"""
1. If the model is still linear in the parameters then it is still considered a linear regression model. This is because the model still learns a linear combination of the input features, but with non-linear transformations applied to some or all of the features.
2. Yes. As demonstrated in the example under the "Adding nonlinear features" section that contains logarithmic and exponential features transformations.
3. If we add non-linear features to a linear classification model, the decision boundary becomes more complicated and can be non-linear. Instead of a flat decision boundary in the original feature space, it becomes an uneven surface in a higher dimensional feature space. This surface can capture the non-linear relationship between the features and the target function.
"""

part4_q3 = r"""
1. Because np.logspace generates values that are spaced logarithmically, which is useful for covering a wide range of magnitudes of lambda. In contrast, np.linspace generates values that are spaced linearly, which can lead to a range that is not suitable for the problem.
2. 60 times. 20 lambda values * 3 values for degree.
"""

# ==============
