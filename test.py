import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils
import testCase

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_paremeters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

# print("----------test update_parameters_with_gd-------------")
# parameters, grads, learning_rate = testCase.update_parameters_with_gd_test_case()
# parameters = update_paremeters_with_gd(parameters, grads, learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches=[]

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((1, m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batche_X = shuffled_X[:, k*mini_batch_size: (k+1)*mini_batch_size]
        mini_batche_Y = shuffled_Y[:, k*mini_batch_size: (k+1)*mini_batch_size]

        mini_batch = (mini_batche_X, mini_batche_Y)
        mini_batches.append(mini_batch)

    if m%mini_batch_size !=0:
        mini_batche_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batche_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batche_X, mini_batche_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# print("----------test random_mini_batches----------")
# X_assess, Y_assess, mini_batch_size = testCase.random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
#
# print("dim of first mini_batch_X:" , mini_batches[0][0].shape)
# print("dim of first mini_batch_Y:" , mini_batches[0][1].shape)
# print("dim of second mini_batch_X:" , mini_batches[1][0].shape)
# print("dim of second mini_batch_Y:" , mini_batches[1][1].shape)
# print("dim of third mini_batch_X:" , mini_batches[2][0].shape)
# print("dim of third mini_batch_Y:" , mini_batches[2][1].shape)

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

    return v

def update_parameters_with_momentun(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads["db" + str(l+1)]

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]

    return parameters, v

# print("----------test update_parameters_with_momentun-----------")
# parameters, grads, v = testCase.update_parameters_with_momentum_test_case()
# update_parameters_with_momentun(parameters, grads, v, beta=0.9, learning_rate=0.01)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[dW1] = " + str(v["dW1"]))
# print("v[db1] = " + str(v["db1"]))
# print("v[dW2] = " + str(v["dW2"]))
# print("v[db2] = " + str(v["db2"]))

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

    return (v,s)

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-np.power(beta1, t))

        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * np.square(grads["db" + str(l+1)])

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(beta2, t))

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)]) / np.sqrt(s_corrected["dW" + str(l+1)] + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)]) / np.sqrt(s_corrected["db" + str(l+1)] + epsilon)

    return (parameters, v, s)
#
# print("------------test update_with_parameters_with_adam-----------")
# parameters, grads, v, s = testCase.update_parameters_with_adam_test_case()
# update_parameters_with_adam(parameters, grads, v, s, t=2)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#
# print("v[dW1] = " + str(v["dW1"]))
# print("v[db1] = " + str(v["db1"]))
# print("v[dW2] = " + str(v["dW2"]))
# print("v[db2] = " + str(v["db2"]))
#
# print("s[dW1] = " + str(s["dW1"]))
# print("s[db1] = " + str(s["db1"]))
# print("s[dW2] = " + str(s["dW2"]))
# print("s[db2] = " + str(s["db2"]))

train_X, train_Y = opt_utils.load_dataset(is_plot=True)

def model(X, Y, layers_dims, optimizer, learning_rate=0.007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):
    L = len(layers_dims)
    costs=[]
    t = 0
    seed = 10

    parameters = opt_utils.initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    else:
        print("optimizer error! exit!")
        exit(1)

    for i in range(num_epochs):
        seed = seed+1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            A3, cache = opt_utils.forward_propagation(minibatch_X, parameters)

            cost = opt_utils.compute_cost(A3, minibatch_Y)

            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)

            if optimizer == "gd":
                parameters = update_paremeters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters = update_parameters_with_momentun(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t+1
                parameters == update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        if i % 100 == 0:
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print(str(i) + "th iteration, cost is " + str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters

layers_dims = [train_X.shape[0], 5, 2, 1]
#parameters = model(train_X, train_Y, layers_dims, optimizer="gd", is_plot=True)
parameters = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum", is_plot=True)

preditions = opt_utils.predict(train_X, train_Y, parameters)
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)