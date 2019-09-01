'''
Name: Sanyukta Sanjay Kate(ssk8153)
      Amit Magar (ajm6745)
Project 2 - This program is the trainMLP program which is responsible for building and training the neural network
            with 1 hidden layers with 5 hidden neurons and 4 output neurons. This program also stored the weights
            in a csv file depending upon the number of epochs given.
'''

import csv
import math
import random
from matplotlib import pyplot as plt

#assign all the global variables
no_hidden_layers = 1  #as given in the problem statement
no_hidden_neurons = 5 #as given in the problem statement
no_output_neurons = 4 #because 4 classes
alpha = 0.01 # as given
no_of_epochs = 1000 #number of epochs fro trainin the MLP. This can be changed
no_bias_nodes = 2 #number of bias nodes
bias_value = 1 #is the bias value which is used during calculating the sigmoid values

def readFile(file_name):
    '''
    This function opens the file and loads the training set
    :param file_name: is the file name of the training dataset
    :return:
    '''

    firstLine = False
    instances = []

    # Open up the csv file
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')

        # for every row in the csv file
        for every_row in readCSV:
            row = []
            # Skip the first line
            if firstLine:
                firstLine = False
                continue
            else:
                for column in every_row:
                    # convert every string value into float
                    row.append(float(column))
                instances.append(row)
    return instances

def build_neural_network(instances):

    global no_hidden_neurons
    global no_output_neurons
    global no_of_epochs
    global no_bias_nodes

    weights = initialize_weights(no_hidden_neurons, 2)
    output_layer_weights = initialize_weights(no_output_neurons, no_hidden_neurons)

    bias_weights = initialize_bias_weights(no_hidden_neurons,1)
    output_layer_bias_weights = initialize_bias_weights(no_output_neurons, 1)

    sse_list = []
    epoch_list = []

    for each_epoch in range(no_of_epochs):
        for each_instance in instances:

            x1 = each_instance[0]
            x2 = each_instance[1]
            y = each_instance[2]
            X = [x1,x2]

            binary_class = findBinaryClass(y)

            #-----------Forward Propogation-------------#

            hidden_layer_sigmoids = calculate_prediction(no_hidden_neurons, weights, bias_weights, x1, x2)

            output_layer_sigmoids = calculate_outputlayer_predictions(no_output_neurons, output_layer_weights, output_layer_bias_weights, hidden_layer_sigmoids)
            #print(len(output_layer_sigmoids))
            #-------------BackPropogation--------------#

            output_layer_weights, weights, output_layer_bias_weights, bias_weights = BackPropogation(binary_class, weights, output_layer_weights, bias_weights, output_layer_bias_weights, hidden_layer_sigmoids, output_layer_sigmoids, X)

            #print(output_layer_sigmoids)
            # find the see by calling the function and appending the result to the sse_list
        sse_list.append(calculate_error(output_layer_sigmoids, instances))
        epoch_list.append(each_epoch)

    plotGraph(sse_list, epoch_list)

    '''
    print(weights)
    print("output_layer_wts")
    print(output_layer_weights)
    print('Bias weights')
    print(bias_weights)
    print('output bias weights')
    print(output_layer_bias_weights)
    '''
    #Write the Weights into the csv files
    write_weights(weights, output_layer_weights, bias_weights, output_layer_bias_weights, no_of_epochs)

def plot_training_samples(instances):
    X_axis_red = []
    Y_axis_red = []

    X_axis_blue = []
    Y_axis_blue = []

    X_axis_green = []
    Y_axis_green = []

    X_axis_yellow = []
    Y_axis_yellow = []

    for each_instance in instances:

        if each_instance[2] == 1:
            X_axis_red.append(each_instance[0])
            Y_axis_red.append(each_instance[1])

        if each_instance[2] == 2:
            X_axis_blue.append(each_instance[0])
            Y_axis_blue.append(each_instance[1])

        if each_instance[2] == 3:
            X_axis_green.append(each_instance[0])
            Y_axis_green.append(each_instance[1])

        if each_instance[2] == 4:
            X_axis_yellow.append(each_instance[0])
            Y_axis_yellow.append(each_instance[1])

    plt.scatter(X_axis_red, Y_axis_red, color='red', label='Bolts')
    plt.scatter(X_axis_blue, Y_axis_blue, color='green', label='Nuts')
    plt.scatter(X_axis_green, Y_axis_green, color='yellow', label='Rings')
    plt.scatter(X_axis_yellow, Y_axis_yellow, color='blue', label='Scrap')

    plt.xlabel('Six-fold rotational symmetry')
    plt.ylabel('Eccentricity')

    plt.legend()
    plt.show()


def calculate_error(predicted_output, instances):

    error=0
    for each_instance in instances:
        #print(each_instance, "each")
        #get the max value of the predicted_output and it's index + 1
        index_max_val = (predicted_output.index(max(predicted_output)))

        #print(index_max_val)
        if index_max_val!=each_instance[2]:
            error+=math.pow((0.0-max(predicted_output)),2)
        #else:
        #    error+=math.pow(1.1-max(predicted_output),2)
    return error


def write_weights(weights, output_layer_weights, bias_weights, output_layer_bias_weights, no_of_epochs):
    '''
    Writes the wts into the csv files depending upon the epoch number.
    :param weights: weights of the input -hidden layer
    :param output_layer_weights: weights of the hidden to output layer
    :param no_of_epochs: the no of epochs given
    :return:  none
    '''

    if no_of_epochs == 0:
        filename='weights0.csv'
    elif no_of_epochs == 10:
        filename='weights10.csv'
    elif no_of_epochs == 100:
        filename ='weights100.csv'
    elif no_of_epochs == 1000:
        filename = 'weights1000.csv'
    else:
        filename = 'weights10000.csv'

    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')

        for row in range(len(weights)):
            writer.writerow((weights[row]))
        for row in range(len(output_layer_weights)):
            writer.writerow(output_layer_weights[row])
        #for row in range(len(bias_weights)):
        writer.writerow(bias_weights)
        writer.writerow(output_layer_bias_weights)

def plotGraph(errors, epochs):
    '''
    This function plots the graph of epochs vs sum of square errors
    :param errors: The sum of square errors
    :param epochs: The number of epochs
    :return: None
    '''
    print("\nPlotting the SSE vs Epoch graph")
    plt.figure()
    plt.plot(epochs, errors, label='Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Sum of Square Errors (SSE)')
    plt.legend()
    plt.title('Epoch vs SSE Graph')
    plt.show()

def BackPropogation(binary_class, weights, output_layer_weights, bias_weights, output_layer_bias_weights, hidden_layer_sigmoids, output_layer_sigmoids, X):
    '''
    :param binary_class: binary representation of the class. Class 2 is represented as [0100]
    :param weights: the input to hidden layer weights
    :param output_layer_weights: the hidden to output layer weights
    :param bias_weights: the input to hidden layer weights
    :param output_layer_bias_weights: the hidden to output layer weights
    :param hidden_layer_sigmoids: Activations of the hidden layer
    :param output_layer_sigmoids: Activations of the output layer (predicted output)
    :return:
    '''
    #-----------Calculating Delta Values-------------#
    output_layer_deltas = [] #length is 4
    hidden_layer_deltas = [] # length is 5

    # for each of the output neuron, calculating the delta value
    for each_neuron in range(no_output_neurons):
        sigmoid_derivative = output_layer_sigmoids[each_neuron] * (1-output_layer_sigmoids[each_neuron])
        error_value = output_layer_sigmoids[each_neuron]-binary_class[each_neuron]
        #error_value = binary_class[each_neuron]-output_layer_sigmoids[each_neuron]
        delta_value = sigmoid_derivative * error_value

        output_layer_deltas.append(delta_value)

    #for each of the hidden neuron, calculating the delta value. Here, we have to use the sigmoid values of the hidden neurons
    for column in range(no_hidden_neurons):
        sigmoid_derivative = hidden_layer_sigmoids[column]*(1-hidden_layer_sigmoids[column])
        sum = 0
        for row in range(no_output_neurons):
            sum+=output_layer_deltas[row]*output_layer_weights[row][column]
        delta_value=sigmoid_derivative*sum
        hidden_layer_deltas.append(delta_value)

    #Note: the deltas for both the bias nodes will be 0

    #------------Updating the Weights-------------#

    #updating the weights from hidden to output layer, that is output layer weights
    for row in range(no_output_neurons):
        alpha_delta = alpha*output_layer_deltas[row]
        for column in range(no_hidden_neurons):
            output_layer_weights[row][column] = output_layer_weights[row][column]-(alpha_delta*hidden_layer_sigmoids[column])

    #updating the weights for input to hidden layer nodes, that is the hidden layer weights
    for row in range(0, no_hidden_neurons):
        alpha_delta = alpha*hidden_layer_deltas[row]
        for column in range(2):
            weights[row][column] = weights[row][column]-(alpha_delta*X[column])


    #updating the weights for the bias nodes
    for row in range(no_output_neurons):
        output_layer_bias_weights[row] = output_layer_bias_weights[row]-(alpha*output_layer_deltas[row]*1)

    for row in range(no_hidden_neurons):
        bias_weights[row] = bias_weights[row]-(alpha*hidden_layer_deltas[row]*1)

    return output_layer_weights, weights, output_layer_bias_weights, bias_weights


def calculate_outputlayer_predictions(no_output_neurons, weights, bias_weights, hidden_layer_sigmoids):

    '''

    :param no_output_neurons: the output layer neurons, so there will be 4 sigmoid values from the output layer
    :param weights: the hidden to output layer weights
    :param bias_weights: the bias wts of hidden to output layer
    :param hidden_layer_sigmoids: the input for the output layer, total 5 values here
    :return:
    '''

    global no_hidden_neurons
    prediction_list=[]
    for each_neuron in range(no_output_neurons):

        z = bias_weights[each_neuron]
        for index in range(0, no_hidden_neurons):
            z += (weights[each_neuron][index]*hidden_layer_sigmoids[index])
        negative_z = (-1) * z
        sigmoid_value = 1/(1+math.exp(negative_z))
        prediction_list.append(sigmoid_value)

    return prediction_list


def calculate_prediction(no_hidden_neurons, weights, bias_weights, x1, x2):

    #There should be 5 sigmoid values of the hidden layer as there are 5 hidden nodes
    prediction_list = []

    for each_neuron in range(no_hidden_neurons):
        # calculate the prediction --- formula ---> 1/1+e^(-(b0+b1*x1+b2*x2)) ---> sigmoid

        z = bias_weights[each_neuron]+((weights[each_neuron][0])*x1)+((weights[each_neuron][1])*x2)
        #print("z ",z)
        negative_z = (-(z))
        sigmoid_value = 1/(1+math.exp(negative_z))
        prediction_list.append(sigmoid_value)

    return prediction_list


def findBinaryClass(class_value):
    '''
    creates a binary list depending upon the class value.
    :param class_value:
    :return:
    '''

    if class_value == 1:
        y = [1,0,0,0]
    elif class_value == 2:
        y = [0,1,0,0]
    elif class_value == 3:
        y = [0,0,1,0]
    else:
        y = [0,0,0,1]

    return y

def initialize_bias_weights(no_neurons, no_bias_nodes):
    '''

    :param no_neurons: number of hidden neurons
    :param no_bias_nodes: number of bias nodes/neurons in the input layer
    :return:
    '''
    bias_weights = [None for x in range(no_neurons)]

    for row in range(0, no_neurons):
        bias_weights[row] = random.uniform(1,-1)

    return bias_weights


def initialize_weights(no_row_neurons, no_column_neurons):
    '''

    :param no_row_neurons: no of hidden neurons
    :param no_column_neurons: no of input layer neurons
    :return: randomly initialzed wts
    '''
    # create a no_hidden_neuron * number_of attributes matrix and randomly initialze wts (here, 5*2 weights matrix)
    weights = [[None for x in range(no_column_neurons)] for y in range(no_row_neurons)]

    for row in range(0, no_row_neurons):
        # random.seed(datetime.now())
        for column in range(0, no_column_neurons):
            weights[row][column] = random.uniform(1,-1)

    #print(weights)
    return weights


def main():
    #print('Enter the file name')
    #filename = input()
    filename = 'train_data.csv'
    instances = readFile(filename)
    build_neural_network(instances)
    plot_training_samples(instances)

if __name__=='__main__':
    main()