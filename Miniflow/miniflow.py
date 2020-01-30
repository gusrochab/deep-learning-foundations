import numpy as np

class Node:
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this node passes values
        self.outbound_nodes = []
        # for each inbound_node, add the current Node as an outbound_node
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)
        self.value = None
        self.gradients = {}

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            self.gradients[self] += grad_cost * 1


class Add(Node):
    def __init__(self, *args):
        Node.__init__(self, args)
        self.value = 0

    def forward(self):
        for node in self.inbound_nodes:
            self.value += node.value
        return self.value


class Mul(Node):
    def __init__(self, *args):
        Node.__init__(self, args)
        self.value = 1
    def forward(self):
        for node in self.inbound_nodes:
            self.value *= node.value
        return self.value


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias  = self.inbound_nodes[2].value
        self.value = np.dot(inputs, weights) + bias
        #return self.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, linear):
        Node.__init__(self, [linear])

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self):
        linear = self.inbound_nodes[0].value
        self.value = self.sigmoid(linear)
        return self.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a
        self.value = np.mean(self.diff**2)
        #return self.value

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff


def topological_sort(feed_dict):
    """
    Sort gneric nodes in topological order using Kahn's Algorithm
    :param feed_dict: A dictionary where the key is a 'Input' node and the value is the respective value feed to that node
    :return: A list of sorted nodes
    """

    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes
    :param output_node: A node in the graph, should be output node (have no outgoing edges)
    :param sorted_nodes: A toplogically sorted list of nodes
    :return: The output node's value
    """

    for node in sorted_nodes:
        node.forward()

    for node in sorted_nodes[::-1]:
        node.backward()

    return output_node.value


def sgd_update(trainable, learning_rate=1e-2):
    """
    Updates the value of eatch trainable with SGD
    :param trainable: A list of Input Nodes representing weights and biases
    :param learning_rate: The learning rate
    :return:
    """
    for n in trainable:
        n.value = n.value - learning_rate * n.gradients[n]


