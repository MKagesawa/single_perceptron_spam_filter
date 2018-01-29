import pickle
import numpy as np
import matplotlib.pyplot as plt

# Train data input sizes
N = [100, 500, 1000, 2000, 4000]

# Loading pickles
test_vector_object = open("test_vector.pkl", "rb")
test_vector = pickle.load(test_vector_object)

train_vector_object = open("train_vector.pkl", "rb")
train_vector = pickle.load(train_vector_object)

validation_vector_object = open("validation_vector.pkl", "rb")
validation_vector = pickle.load(validation_vector_object)

test_01_object = open("test_01.pkl", "rb")
test_01 = pickle.load(test_01_object)

validation_01_object = open("validation_01.pkl", "rb")
validation_01 = pickle.load(validation_01_object)

train_01_object = open("train_01.pkl", "rb")
train_01 = pickle.load(train_01_object)

freq_list_object = open("freq_list.pkl", "rb")
freq_list = pickle.load(freq_list_object)

t_v_vector_object = open("train+validation.pkl", "rb")
t_v_vector = pickle.load(t_v_vector_object)

t_v_01_object = open("train+validation_01.pkl", "rb")
t_v_01 = pickle.load(t_v_01_object)

# Tests the error rate of a model
def perceptron_error(w, vector, label):

    matrix = np.array([np.array(a) for a in vector])
    error = 0

    for i in range(len(matrix)):
        result = np.dot(w, matrix[i])
        # Incorrectly Classified
        if np.multiply(result, label[i]) <= 0:
            error += 1

    # Find the fraction of input incorrectly classified
    fraction = error/len(matrix)

    return fraction


# Assumes the input data is linearly separable
# word_list: list of words that appear at least on 25 seperate emails
# max_pass: Set 'stop' val for slicing input vector when iterating
# max_iter: Setting max number of times the entire vector input can be iterated for training
def perceptron_train(vector, label, word_list, max_pass, max_iter):

    # Make list to NumPy array
    matrix = np.array([np.array(a) for a in vector])

    # Final classification filter
    w = np.zeros(len(word_list), dtype=int)

    # Number of updates(mistakes) performed
    k = 0

    # Number of passes through the data
    iteration = 0

    while (perceptron_error(w, vector, label) != 0) and (iteration < max_iter):
        iteration += 1
        for i in range(max_pass):
            result = np.dot(w, matrix[i])
            # Correctly classified
            if result * label[i] > 0:
                w = w
            else:
                k += 1
                w += np.multiply(label[i], matrix[i])

    return w, k, iteration


"""
# Pickle trained weight
w_object = open("weight.pkl", "wb")
pickle.dump(perceptron_train(train_vector, train_01, freq_list), w_object)
w_object.close()
"""

# Load trained weight
weight_object = open("weight.pkl", "rb")
weight = pickle.load(weight_object)

# Train with train data vector and train data label, then output error fraction testing against validation data set
# print(perceptron_error(perceptron_train(train_vector, train_01, freq_list, 4000, 30)[0], validation_vector, validation_01))

# Finding the 8 words with most positive weights and 8 most negative words
def most_positive_negative(w, freq_words):
    # w[0] because input w from perceptron_train() is a list containing w, k, iteration but we only need w
    sorted = np.argsort(weight[0])    # Index of sorted array
    positive_index = []
    negative_index = []
    for i in range(8):
        negative_index.append(sorted[i])
    for j in range(8):
        positive_index.append(sorted[-j - 1])

    # Words most likely spam (highest weights)
    positive_list = []
    # Words most likely not spam (lowest weights)
    negative_list = []

    # Find the corresponding word from freq_words list
    for pos in positive_index:
        positive_list.append(freq_words[pos])

    for neg in negative_index:
        negative_list.append(freq_words[neg])

    return positive_list, negative_list

def plot(train_vector, train_label, N, word_list, validation_vector, validation_label):
    # Using only first N row of training data, train, and plot corresponding validation error
    plot_y_axis = []
    for n in N:
        plot_y_axis.append(perceptron_error(perceptron_train(train_vector[:n], train_label, word_list, n)[0], validation_vector, validation_label))


    #Round Red plots
    plt.plot(N, plot_y_axis, 'ro')
    plt.xlabel('Input Size N')

    # Plotting iterations it takes to converge with different N inputs
    iterations = []
    for n in N:
        iterations.append(perceptron_train(train_vector[:n], train_label, word_list, n)[2]) #[2] from returned (w, k, iteration)
    # Square Blue plots
    plt.plot(N, iterations, 'sb')
    plt.show()


#plot(train_vector, train_01, N, freq_list, validation_vector, validation_01)