from Miniflow2.miniflow import *
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample


data = load_boston()

# Load data
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Newral network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()
l1 = Linear(X, W1, b1)
s = Sigmoid(l1)
l2 = Linear(s, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 10000

# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size

sorted_nodes = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print('Total number of examples = {}'.format(m))

for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Set the value of X any Inputs
        X.value = X_batch
        y.value = y_batch

        forward_and_backward(cost, sorted_nodes)
        sgd_update(trainables, 0.001)
        loss = sorted_nodes[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss/steps_per_epoch))





###################################################
#Backpropagation
'''
X, W, b = Input(), Input(), Input()
y = Input()
linear = Linear(X, W, b)
sigmoid = Sigmoid(linear)
cost = MSE(y, sigmoid)

X_ = np.array([[-1., -2.], [-1., -2.]])
W_ = np.array([[2.], [3.]])
b_ = np.array([-3.])
y_= np.array([1, 2])

feed_dict = {
    X: X_,
    y: y_,
    W: W_,
    b: b_
}

sorted_nodes = topological_sort(feed_dict)
forward_and_backward(cost, sorted_nodes)
gradients = [t.gradients[t] for t in [X, y, W, b]]
print(gradients)
'''

###################################################
#Cost
'''
y, a = Input(), Input()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}

sorted_nodes = topological_sort(feed_dict)
print(forward_pass(cost, sorted_nodes))
'''

####################################################
# Sigmoid
'''
X, W, b = Input(), Input(), Input()
linear = Linear(X, W, b)
sigmoid = Sigmoid(linear)

X_ = np.array([[-1, -2], [-1, -2]])
W_ = np.array([[2, -3], [2, -3]])
b_ = np.array([-3, -5])

feed_dict = {
    X: X_,
    W: W_,
    b: b_
}

sorted_nodes = topological_sort(feed_dict)

print(forward_pass(sigmoid, sorted_nodes))
'''

####################################################
# Linear
'''
inputs, weights, bias = Input(), Input(), Input()

linear = Linear(inputs, weights, bias)
feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}
sorted_nodes = topological_sort(feed_dict)

print(forward_pass(linear, sorted_nodes))
'''

####################################################
# Add
'''
x, y, z, w = Input(), Input(), Input(), Input()
add = Add(x, y, z, w)
feed_dict = {x: 10, y: 20, z:5, w:10}
#mul = Mul(x, y, z, w)
#feed_dict = {x: 1, y: 2, z:3, w:4}

sorted_nodes = topological_sort(feed_dict)

def forward_pass(output_node, sorted_nodes):
    for node in sorted_nodes:
        node.forward()
    return output_node.value


print(forward_pass(add, sorted_nodes))
'''