import numpy as np
# Relu
def relu(x):
    if x > 0:
        return x
    else:
        return 0


activation_func = np.vectorize(relu)


def derivative(x):
    if x > 0:
        return 1
    else:
        return 0

derivative_func = np.vectorize(derivative)


class Weight:
    def __init__(self, m, n):
        self.sigma = 0.2  # 0.1 # 0.3 #0.2
        self.mu = 0  # 0.5
        self.m = m
        self.n = n
        self.val = self.sigma * np.random.randn(m + 1, n) + self.mu

    def normalize(self):
        self.val = self.val / self.val.sum()

class Bias:
    def __init__(self, n):
        self.val = np.random.randn(1, n)


class Layer:
    def __init__(self, n, id=None, output_ind=False):
        self.val = None
        self.id = id
        self.input_weight = None
        self.output_weight = None
        self.input_bias = None
        self.output_bias = None
        self.pre_layer = None
        self.next_layer = None
        self.n = n
        self.delta_weight = None
        self.output_ind = output_ind

    def full_connect(self, other):
        m = self.n
        n = other.n
        weight = Weight(m, n)
        bias = Bias(n)
        self.output_weight = weight
        self.output_bias = bias
        other.input_weight = weight
        other.input_bias = bias

        self.next_layer = other
        other.pre_layer = self

    def stats(self):
        print("n:", self.n)
        if self.output_weight:
            print('Weight:', self.output_weight.val.shape)
        print()

    def pull_in(self, input_list):
        self.val = np.array([input_list, ])

    def forwarding(self):
        val = self.val
        val = np.append(val, [1])
        print('val:', val.shape)
        print('weight:', self.output_weight.val.shape)
        val = val.dot(self.output_weight.val) + self.bias.val

        try:
            if self.next_layer:
                if not self.next_layer.output_ind:
                    val = activation_func(val)
                    self.next_layer.val = val
                else:
                    self.next_layer.val = val
        except:
            raise

    def backwarding(self, error_term, learning_rate):
        pre = self.pre_layer
        # K1 = pre.output_weight.val
        # while (pre.pre_layer):
        #     pre = pre.pre_layer
        #     K1 = pre.output_weight.val.dot(K1)
        #
        # pre_val = np.append(pre.val, [1])
        # K1 = pre_val.dot(K1)

        # Input
        K1 = pre.val

        # Chain
        next_layer = self.next_layer
        if next_layer.output_weight:
            K2 = next_layer.output_weight.val
            while (next_layer.next_layer.output_weight):
                next_layer = next_layer.next_layer
                K2 = K2.dot(next_layer.output_weight.val)
        else:
            K2 = np.array([[1]])

        chain = K2

        cur_val = self.val

        # g'(f(w,x,b))
        g_derivative = derivative_func(cur_val)

        # f'(w,x,b)
        f_derivative = K1


        # Delta Weight
        delta_weight = (-1) * error_term * g_derivative * f_derivative * chain

        # if not self.output_ind:
        #     derivative = derivative * K3.T

        if type(self.delta_weight) != type(None):
            self.delta_weight += delta_weight
        else:
            self.delta_weight = delta_weight

        # Delta Bias
        delta_bias








    def update_weights(self):
        if type(self.delta_weight) != type(None):
            self.output_weight.val += self.delta_weight
            self.delta_weight = None


class Network:
    def __init__(self, input_layer, output_layer):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.learning_rate = 0.01

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    def batch_forwarding(self):
        cur = self.input_layer
        while (cur.next_layer):
            cur.forwarding()
            cur = cur.next_layer

    def get_error_term(self, ground_truth_list):
        error = self.output_layer.val - ground_truth_list
        return error

    def batch_backwarding(self, error_term):
        cur = self.input_layer.next_layer
        while (cur and cur.output_weight):
            cur.backwarding(error_term, self.learning_rate)
            cur = cur.next_layer

    def one_train(self, x, y):
        input_layer.pull_in(x)
        self.batch_forwarding()
        error_term = self.get_error_term(y)
        self.batch_backwarding(error_term)

    def batch_update_weights(self):
        cur = self.input_layer.next_layer
        while (cur and cur.output_weight):
            cur.update_weights()
            cur = cur.next_layer

    def batch_train(self, X, Y, batch_size=20):
        i = 0
        for i, (x, y) in enumerate(zip(X, Y)):
            self.one_train(x, y)
            if i % batch_size == 0:
                self.batch_update_weights()
        self.batch_update_weights()

    def predict(self, x):
        input_layer.pull_in(x)
        self.batch_forwarding()
        return self.output_layer.val


if __name__=="__main__":
    input_layer = Layer(1, 'in')
    l1 = Layer(10, 'l1')
    l2 = Layer(10, 'l2')
    output_layer = Layer(1, 'out', output_ind=True)

    input_layer.full_connect(l1)
    l1.full_connect(l2)
    l2.full_connect(output_layer)

    nk = Network(input_layer, output_layer)

    X = []
    Y = []
    for i in range(20000):
        x = np.array([np.random.rand(), ])
        if x > 0.5:
            y = 1
        else:
            y = 0

        X.append(x)
        Y.append(y)

    nk.set_learning_rate(0.01)
    nk.batch_train(X, Y)

    for i in range(10):
        x = np.array([np.random.rand(), ])
        if x > 0.5:
            y = 1
        else:
            y = 0
        y_predict = nk.predict(x)

        print('x:', x)
        print('y_real:', y)
        print('y_predict:', y_predict)
        print()