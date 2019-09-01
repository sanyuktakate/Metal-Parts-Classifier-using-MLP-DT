'''
Name: Sanyukta Sanjay Kate and AMit Magar
executeMLP is the file which loads the saved wts and then tests the neural network using the test dataset and the wts saved.
Also generates the graph of the different classes with the decision boundary. ALso the profit is found
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
no_of_epochs = 0 #number of epochs fro trainin the MLP. This can be changed
no_bias_nodes = 2 #number of bias nodes
bias_value = 1 #is the bias value which is used during calculating the sigmoid values
#The profit matrix is given
given_profit_matrix =      [[20, -7, -7, -7],
                           [-7, 15, -7, -7],
                           [-7, -7, 5, -7],
                           [-3, -3, -3, -3]]

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

def read_Weights():

    #Reads the weights saved in the csv file
    #:return:
    temp_weights = []

    with open('weights1000.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')

        for each_row in readCSV:
            temp_weights.append(each_row)

    weights = []
    for row in range(len(temp_weights)):
        if(row<5):
            weights.append(temp_weights[row])
        else:
            break

    output_layer_weights = []

    for r1 in range(row, len(temp_weights)-2):
        output_layer_weights.append(temp_weights[r1])

    bias_weights=[]
    for r2 in range(r1, len(temp_weights)-1):
        bias_weights.append(temp_weights[r2])

    output_bias_weights=[]
    output_bias_weights.append(temp_weights[-1])

    #print(bias_weights)
    #print(output_bias_weights)
    new_bias_wts=[]
    for row in range(0, len(bias_weights)):
        for column in range(0, len(bias_weights[0])):
            bias_weights[row][column]=float(bias_weights[row][column])
            new_bias_wts.append(bias_weights[row][column])

    new_output_bias_wts=[]
    for row in range(0, len(output_bias_weights)):
        for column in range(0, len(output_bias_weights[0])):
            output_bias_weights[row][column]=float(output_bias_weights[row][column])
            new_output_bias_wts.append(output_bias_weights[row][column])

    for row in range(0, len(weights)):
        for column in range(0, len(weights[0])):
            weights[row][column] = float(weights[row][column])

    for row in range(0, len(output_bias_weights)):
        for column in range(0, len(output_layer_weights[0])):
            output_layer_weights[row][column]=float(output_layer_weights[row][column])


    return weights, output_layer_weights, new_bias_wts, new_output_bias_wts


def build_neural_network(instances):

    global no_hidden_neurons
    global no_output_neurons
    global no_of_epochs
    global no_bias_nodes


    weights, output_layer_weights, bias_weights, output_layer_bias_weights = read_Weights()

    #for each_epoch in range(no_of_epochs):
    predicted_class = []

    for each_instance in instances:

        x1 = each_instance[0]
        x2 = each_instance[1]
        y = each_instance[2]
        X = [x1,x2]
        #print(x1, "aba")

        binary_class = findBinaryClass(y)

        #-----------Forward Propogation-------------#

        hidden_layer_sigmoids = calculate_prediction(no_hidden_neurons, weights, bias_weights, x1, x2)

        output_layer_sigmoids = calculate_outputlayer_predictions(no_output_neurons, output_layer_weights, output_layer_bias_weights, hidden_layer_sigmoids)
        predicted_class.append(find_class(output_layer_sigmoids))

    #print(output_layer_sigmoids)
    #print(predicted_class)
    accuracy, confusion_matrix = find_accuracy(instances, predicted_class)
    find_profit(confusion_matrix)

def find_profit(confusion_matrix):
    '''
    This func finds the profit using the given matrix (global matrix)
    :param confusion_matrix: confusion matrix whih is displayed on the console
    :return:
    '''

    global given_profit_matrix

    profit = 0
    for row in range(4):
        for column in range(4):
            profit += confusion_matrix[row][column]*given_profit_matrix[row][column]

    print("\nThe Profit Is: ", profit)

def find_accuracy(instances, predicted_class):
    '''
    finds the accuracy of the system
    :param instances: test dataset
    :param predicted_class: predicted class list
    :return:
    '''
    index = 0
    class1_number_of_correctly_classified = 0
    class1_number_of_wrongly_classified = 0
    class2_number_of_correctly_classified = 0
    class2_number_of_wrongly_classified = 0
    class3_number_of_correctly_classified = 0
    class3_number_of_worongly_classified = 0
    class4_number_of_correctly_classified = 0
    class4_number_of_worongly_classified = 0

    confusion_matrix = [[0 for row in range(4)] for column in range(4)]

    #finding the accuracy and corrrectly and incorrectly classified
    #predicted is the rows and the actual is the column

    for each_instance in instances:
        if each_instance[2] == 1:
            if each_instance[2] == predicted_class[index]:
                class1_number_of_correctly_classified+=1
                confusion_matrix[0][0]+=1
            else:
                class1_number_of_wrongly_classified+=1
                confusion_matrix[predicted_class[index]-1][0]+=1


        if each_instance[2] == 2:
            if each_instance[2] == predicted_class[index]:
                class2_number_of_correctly_classified += 1
                confusion_matrix[1][1] += 1
            else:
                class2_number_of_wrongly_classified += 1
                confusion_matrix[predicted_class[index]-1][1] += 1


        if each_instance[2] == 3:
            if each_instance[2] == predicted_class[index]:
                class3_number_of_correctly_classified += 1
                confusion_matrix[2][2] += 1
            else:
                class3_number_of_worongly_classified += 1
                confusion_matrix[predicted_class[index]-1][2] += 1

        if each_instance[2] == 4:
            if each_instance[2] == predicted_class[index]:
                class4_number_of_correctly_classified += 1
                confusion_matrix[3][3] += 1
            else:
                class4_number_of_worongly_classified += 1
                confusion_matrix[predicted_class[index]-1][3] += 1
        index+=1
    accuracy = ((class1_number_of_correctly_classified+class2_number_of_correctly_classified+class3_number_of_correctly_classified+class4_number_of_correctly_classified)/len(instances))*100
    print("The % of accuracy is:", accuracy, "%")

    row_total = []
    column_total = []

    for row in range(4):
        sum_row = 0
        sum_column = 0
        for column in range(4):
            sum_row += confusion_matrix[row][column]
            sum_column += confusion_matrix[column][row]
        row_total.append(sum_row)
        column_total.append(sum_column)

    print("\n--------------The Confusion Matrix----------------")
    count = 0
    print("                           Actual Class       Total")

    complete_total = 0
    index=0

    for row in confusion_matrix:
        if count == 1:
            print("Predicted Class  class ",(index+1),"  ", row, "    = ",row_total[index] )
        else:
            print("                 class ",(index+1),"  ", row, "    = ",row_total[index])
        complete_total += row_total[index]
        count+=1
        index+=1


    print("--------------------------------------------------")
    print("                 Total =    ", column_total,"    =", complete_total)

    return accuracy, confusion_matrix

def generate_decision_boundary(instances):
    '''

    :param instances:  the test dataset
    :return: none
    '''
    #red, blue, green, yellow

    #red is class 1
    #blue is class 2
    #green is class 3
    #yellow is class 4

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

    x_axis = 0
    y_axis = 0

    plotting_instances = []
    step = 0.001

    while x_axis<=1:
        while y_axis<=1:
            #start plotting
            plotting_instances.append([x_axis,y_axis])
            y_axis+=step
        y_axis = 0
        x_axis+=step

    #print(plotting_instances)

    #now pass all these values through the classification algo
    class1_X_list, class1_Y_list, class2_X_list, class2_Y_list, class3_X_list, class3_Y_list, class4_X_list, class4_Y_list = prediction_for_plotting(plotting_instances)

    plt.scatter(class1_X_list,class1_Y_list, color='magenta', label='Class 1')
    plt.scatter(class2_X_list,class2_Y_list, color='orange', label='Class 2')
    plt.scatter(class3_X_list,class3_Y_list, color='purple', label='Class 3')
    plt.scatter(class4_X_list,class4_Y_list, color='gray', label='Class 4')

    plt.scatter(X_axis_red, Y_axis_red, color = 'red', label = 'Bolts')
    plt.scatter(X_axis_blue, Y_axis_blue, color = 'green', label = 'Nuts')
    plt.scatter(X_axis_green, Y_axis_green, color = 'yellow', label = 'Rings')
    plt.scatter(X_axis_yellow, Y_axis_yellow, color = 'blue', label = 'Scrap')

    plt.xlabel('Six-fold rotational symmetry')
    plt.ylabel('Eccentricity')

    plt.title('Decision Boundary Graph')
    plt.legend()
    plt.show()

def prediction_for_plotting(plotting_instances):
    '''
    using this function for predicting the values for x and y values obtained from the above function (ie, generate_decision_boundary)
    :param plotting_instances:
    :return: different class list segregated using the classes and the X and Y values
    '''

    global no_hidden_neurons
    global no_output_neurons

    predicted_class = []

    # weights = [[4.183498772187368, 3.8002995213712745], [4.218150860524553, 3.161979872959181], [5.070040651181185, 3.4583760595351785], [3.6506109966373534, 4.77843584778913], [4.036666810322284, 4.273749628542261]]
    # output_layer_weights = [[-4.70020126173716, -4.377830021923376, -5.165398005963075, -5.122897742261428, -4.472994525503201], [-3.9971387862865835, -2.791627221424008, -4.2724058144473736, -3.916138644912728, -4.124388329177065], [-3.127647999377898, -3.069901327043883, -4.551413975070307, -4.100988422915991, -2.9324934489327728], [-6.080590153769306, -6.492891448461257, -6.20238383178612, -5.707648805277463, -6.24500984080201]]

    weights, output_layer_weights, bias_weights, output_layer_bias_weights = read_Weights()

    #bias = initialize_bias(no_hidden_neurons)

    class1_X_list = []
    class2_X_list = []
    class3_X_list = []
    class4_X_list = []

    class1_Y_list = []
    class2_Y_list = []
    class3_Y_list = []
    class4_Y_list = []

    for each_instance in plotting_instances:

        # for each instance
        x1 = each_instance[0]
        x2 = each_instance[1]
        #print("aa ", x1)

        #class_no = find_class(output_layer_predicted_outputlist)
        hidden_layer_sigmoids = calculate_prediction(no_hidden_neurons, weights, bias_weights, x1, x2)

        output_layer_sigmoids = calculate_outputlayer_predictions(no_output_neurons, output_layer_weights,
                                                                  output_layer_bias_weights, hidden_layer_sigmoids)
        class_no = find_class(output_layer_sigmoids)
        #predicted_class.append(find_class(output_layer_sigmoids))

        if class_no is 1:
            # append x1 and x2 to the class1list
            class1_X_list.append(x1)
            class1_Y_list.append(x2)

        elif class_no is 2:
            class2_X_list.append(x1)
            class2_Y_list.append(x2)

        elif class_no is 3:
            class3_X_list.append(x1)
            class3_Y_list.append(x2)

        else:
            class4_X_list.append(x1)
            class4_Y_list.append(x2)
    return class1_X_list, class1_Y_list, class2_X_list, class2_Y_list, class3_X_list, class3_Y_list, class4_X_list, class4_Y_list


def find_class(predicted_output):
    '''
    finds the class value of the predicted values
    :param predicted_output:
    :return:
    '''

    max_value = max(predicted_output)
    max_index = predicted_output.index(max_value)

    return max_index+1

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
            #print("Here",hidden_layer_sigmoids[index])
            #print("wts here",weights[each_neuron][index])
            z += (float(weights[each_neuron][index])*float(hidden_layer_sigmoids[index]))
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
        negative_z = (-1*(z))
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
    return weights


def main():
    #print('Enter the file name')
    #filename = input()
    filename = 'test_data.csv'
    instances = readFile(filename)
    build_neural_network(instances)
    generate_decision_boundary(instances)

if __name__=='__main__':
    main()