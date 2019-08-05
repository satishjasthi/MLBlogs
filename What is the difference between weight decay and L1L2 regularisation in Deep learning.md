

# What is the difference between weight decay and L1/L2 regularisation in Deep learning?

Weight decay and  L1/L2 regularisation both methods are used to avoid model overfitting so that the model can generalize better on real data instances. However, there is a minute difference in the way these two methods achieve regularisation methods.

The cost function of a neural network in a simplified manner can be defined as

$$ J(w) = -\frac{1}{m} * \sum(L) $$

ie the average of loss value($L$) computed over each of $m$ data points.

### Regularisation:

Using L1/L2 regularisation redefines the loss function of the network as

for l2
$$ J(w) = -\frac{1}{m} * \sum(L) + \frac{\lambda}{2} * \sum w^2 $$ ..(1)

for l1
$$J(w) = -\frac{1}{m} * \sum(L) + \frac{\lambda}{2} * \sum |w|$$..(2)

where $\sum |w|$ and $\sum w^2$ represents the sum of weights or squared weights from all the
layers of the neural network and $\frac{\lambda}{2}*\sum |w| / w^2$ is the **penalty term** with $\lambda$ value talking value of order 0.005.

Generally, when the network overfits to the training data it tries to memorize the data features by providing higher importance for a small set of features which are repeating in training data to evaluate the target label/value rather than equally considering all features and then evaluating the target label/value. Because of this some of the features get higher weight values compared to others by a significant margin.

Under these conditions using regularisation as shown in eq 1 and 2 adds the penalty term which in turn increases the overall cost function value when weight values increases, however since we are using backpropagation to optimize the cost function, it will forces the network to reduce the weight value to a smaller range which in turn reduces the cost function value.



### Weight decay:

Weight decay also tries to regularise the network but in a slightly different manner.

After every iteration during training, the network weights are updated as shown below,

$W_{new} = W_{old} - \delta W $

where $\delta W $ is calculated by taking the partial derivative of $W$ with respect to cost function $J(w)$ and $W_{old}$ is the weight value from the previous iteration.

Since the whole point of regularisation is to bring down the weight values of a network to a smaller range and avoid overfitting, in weight decay after the weight update we perform

$$W_{new} = W_{new} * 0.98 $$

This reduces the weight value by 2% after every iteration, so as the network is trained for a larger number of iterations, the network's weights will automatically fall under a smaller range.

