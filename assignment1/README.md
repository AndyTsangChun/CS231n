Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Winter 2016.

Q1. KNearestNeighbor
- Calculate the L2 distance between test example and training example
- d2(I1,I2)=√ ∑p(I_p1−I_p2)^2
No Loop:
- Remeber (a-b)^2 = a^2 - 2ab + b^2
Predict:
- Sort the distance array by numpy.argsort
- The function return the index of the sorted array
- And get the first k index out so you can find the label of the corresponding index from y_train
- Try numpy.bincount to count the numbers of each label
- Finally, try numpy.argmax to get the highest, it can break ties by returning the smaller label

Q2. SVMLossNaive
- Lost Function: ∑j≠y_i[max(0,w_T_jx_i−w_T_y_ix_i+ delta)]
dW(slope of W), j is actual output, y is expected output
For j =/= y:
- update W of class j by X[i], iff (score_j - score_y + delta > 0)
For j == y(correct class):
- I = ∑j≠y_i (score_j - score_y + delta > 0)
- update W of class j by -I*X[i], I is number of not margin satisfied class
Vectorized:
- numpy.arange will generate integer array within the interval
- e.g arange(10) will return [0:9], arange(3,5) return [3,4]
- Calculate a mask/scale, scale of each weight respect to each class 

LinearClassifier
- Update Rule: weights = weights - step_size * weights_grad
- y = theta_T * X (<< not vectorize presentation)
Stochastic Gradient Descent:
- Use two for loop to loop over each learning rates and regularization value
- Use numpy.mean to get the accuracy, True = 1 False = 0 
- record the svm model with highest accuracy

Q3. Softmax
- Remember to use numpy.log10 not just numpy.log its has different base
- Its quite similar to SVM but it required to exp all the "scores" and then normalize them
- Probability is return instead of just scores
- From cs231n notes, remember to shift the whole f vectir before exp, prevent blowup

Q4. Two-Layer NN
- y = Theta.T * X 
- cannot use previous softmax class since the reg include w1 and w2

