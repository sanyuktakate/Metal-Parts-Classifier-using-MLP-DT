"""
Author : Amit Magar, Sanyukta Kate
Revision 1.0
"""

import pickle

from trainDT import tree, readCSV, generate_decision_boundary, find_accuracy

PROFIT_MATRIX =            [[20, -7, -7, -7],
                           [-7, 15, -7, -7],
                           [-7, -7, 5, -7],
                           [-3, -3, -3, -3]]


def find_profit(confusion_matrix):
    '''
    This func finds the profit using the given matrix (global matrix)
    :param confusion_matrix: confusion matrix whih is displayed on the console
    :return:
    '''

    global PROFIT_MATRIX

    profit = 0
    for row in range(4):
        for column in range(4):
            profit += confusion_matrix[row][column]*PROFIT_MATRIX[row][column]

    print("\nThe Profit Is: ", profit)

def PrintTree(node,depth):

    if node is not None:
        if(not node.leaf):
            print(" "*depth,"Split Attribute : ",node.splitAttr," Split Value : ",node.splitvalue)
        else:
            print("   "*depth,"Leaf Node Category : ",node.category)

        PrintTree(node.left,depth+2)
        PrintTree(node.right, depth + 2)

def createTree(node,input):
    node = pickle.load(input)
    if (node.left is not None):
        node.left = createTree(node.left, input)
    if (node.right is not None):
        node.right = createTree(node.right, input)
    return node

def classify_(node,example):
    for item in example:
        predict=classify(node,item)
        print("Actual Class : ",item[2])
        print("Predicted Class : ",predict)

def classify(node,example):

    while not node.leaf :
        #node.splitAttr,node.splitvalue
        if(example[node.splitAttr]>node.splitvalue):
            node=node.right
        else:
            node=node.left
    return node.category

def main():
    dfile_name=input('Enter decision tree filename')
    testfile=input("Enter testing file name")


    node=tree()
    with open(dfile_name, 'rb') as file:
        node=createTree(node,file)
    print("----------------------------Decision Tree----------------------------")
    PrintTree(node,0)
    print("-------------------------------Summary-------------------------------")
    list=readCSV(testfile)
    accuracy, matrix = find_accuracy(node, list)
    find_profit(matrix)
    generate_decision_boundary(list,node)


if __name__ == '__main__':
    main()