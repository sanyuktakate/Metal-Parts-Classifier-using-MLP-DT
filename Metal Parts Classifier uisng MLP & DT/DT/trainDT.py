"""
Author : Amit Magar, Sanyukta Kate
Revision 1.0
"""

import math
import operator
import pickle
from matplotlib import pyplot as plt
import queue
import numpy as np

class tree:

    __slots__ = "left","right","category","leaf","splitvalue","splitAttr","categoryCount"

    def __init__(self):
        """
        This Class Represents Node of Tree
        """
        self.left=None
        self.right=None
        self.category=None
        self.leaf=False
        self.splitAttr=None
        self.splitvalue=None
        self.categoryCount=None

    def setLeft(self,childNode):
        self.left=childNode

    def setRight(self,childNode):
        self.right=childNode

    def setCategory(self,category):
        self.category=category
        self.leaf=True

    def setSplit(self,index,value):
        self.splitAttr=index
        self.splitValue=value

    def minHeight(self,node):
        if(node is not None):
            return min(node.minHeight(node.left),node.minHeight(node.right))+1;
        else:
            return 0

    def maxHeight(self,node):
        if(node is not None):
            return max(node.maxHeight(node.left),node.maxHeight(node.right))+1;
        else:
            return 0

    def numberofNode(self,node):
        if(node is not None):
            return 1+node.numberofNode(node.left)+node.numberofNode(node.right)
        else:
            return 0;

    def averageHeight(self,node):
        height=-1;
        que=list()
        temp=list()
        avgHeight=0
        que.append(node)
        while len(que) is not 0:
            height+=1
            count=0
            while len(que) is not 0:
                temp.append(que.pop())
                count=+1
            avgHeight+= 0 if height==0 else (count/height)
            while len(temp) is not 0:
                item=temp.pop()
                if(item.left is not None):
                    que.append(item.left)
                if (item.right is not None):
                    que.append(item.right)
        return avgHeight



    def numberofLeaf(self,node):
        if(node.left is None and node.right is None):
            return 1
        else:
            return node.numberofLeaf(node.left)+node.numberofLeaf(node.right)



def classify(node,example):

    while not node.leaf :
        #node.splitAttr,node.splitvalue
        if(example[node.splitAttr]>node.splitvalue):
            node=node.right
        else:
            node=node.left
    return node.category

def readCSV(fileName):

    fhandler=open(fileName,'r')
    opList=list()

    for line in fhandler:
        holder=line.strip().split(',')
        #print(holder)
        opList.append([float(holder[0]),float(holder[1]),int(holder[2])])
    return opList


def categorize(examples):
    """
    This will count of different categories for given examples
    :param examples: list of examples
    :return: list of category count
    """
    categoryList=[0,0,0,0]

    for item in examples:
        categoryList[item[2]-1]+=1
    return categoryList


def entropy(examples):

    catList=categorize(examples)
    odds=list()
    entropyV=0

    for category in catList:
        val=sum(catList)
        if(val>0):
            odds.append((category/val))

    for i in  range(len(odds)):
        if(odds[i]>0):
            entropyV+=odds[i]*math.log2(odds[i])

    return -entropyV


def split(value,feature,examples):

    left,right=list(),list()
    for i in range(len(examples)):
        if(examples[i][feature]>value):
            right.append(examples[i])
        else:
            left.append(examples[i])

    return left,right



def infoGain(examples,parentEntropy):

    informationGain=[[] for i in range (len(examples[0])-1)]


    for feature in range(len(examples[0])-1):
        examples.sort(key=operator.itemgetter(feature))
        #print(examples)

        for index in range(len(examples)-1):
            binarySpiltVal=(examples[index][feature]+examples[index+1][feature])/2
            left,right=[],[]

            for index1 in range(len(examples)):
                if(examples[index1][feature]>binarySpiltVal):
                    right.append(examples[index1])
                else:
                    left.append(examples[index1])
            leftentropy=entropy(left)
            rightentropy=entropy(right)

            remainder=((len(left)/len(examples))*leftentropy)\
                      +((len(right)/len(examples))*rightentropy)

            informationGain[feature].append((binarySpiltVal,parentEntropy-remainder))
    return informationGain

def plurality(examples):
    cat_list=categorize(examples)
    return cat_list.index(max(cat_list))

def isSameClass(examples):

    flag=True
    for i in range(len(examples)-1):
        if examples[i][-1]!=examples[i+1][-1]:
            flag=False
            break
    return flag


def findHighestInfoGain(infoTable):
    maxE1=0
    featureVal2=0
    featureVal1=0
    #print(infoTable)
    for i in range(len(infoTable[0])):
        if(infoTable[0][i][1]>maxE1):
            maxE1=infoTable[0][i][1]
            featureVal1=infoTable[0][i][0]
    maxE2 = 0
    for i in range(len(infoTable[1])):
        if (infoTable[1][i][1] > maxE2):
            maxE2 = infoTable[1][i][1]
            featureVal2 = infoTable[1][i][0]

    if maxE1>maxE2:
        return 0,featureVal1
    else:
        return 1,featureVal2



def PrintTree(node,depth):

    if node is not None:
        if(not node.leaf):
            print(" "*depth,"Split Attribute : ",node.splitAttr," Split Value : ",node.splitvalue,"Different category node at Node : ",node.categoryCount )
        else:
            print("   "*depth,"Different category examples at Leaf Node : ",node.categoryCount ,"Leaf Node Category : ",node.category)

        PrintTree(node.left,depth+2)
        PrintTree(node.right, depth + 2)


def writeTree(node,file):
    
    if node is not None:
        pickle.dump(node, file, pickle.HIGHEST_PROTOCOL)
        writeTree(node.left, file)
        writeTree(node.right, file)
        
    

def chisquare(node):


    nodeCount=0
    leftNodeCount=0
    rightNodeCount=0
    deviation = 0

    for count in node.categoryCount:
        nodeCount+=count

    #print(node.left.categoryCount)
    for count in node.left.categoryCount:
        leftNodeCount+=count

    for count in node.right.categoryCount:
        rightNodeCount += count

    expected_cost_left=[count*(leftNodeCount/nodeCount) for count in node.categoryCount]
    expected_cost_right=[count*(rightNodeCount/nodeCount) for count in node.categoryCount]

    for iter in range(len(node.left.categoryCount)):
        deviation+= 0 if expected_cost_left[iter]==0 else math.pow((node.left.categoryCount[iter]-expected_cost_left[iter]),2)/expected_cost_left[iter]


    for iter in range(len(node.right.categoryCount)):
        deviation+=0 if expected_cost_right[iter]==0 else math.pow((node.right.categoryCount[iter]-expected_cost_right[iter]),2)/expected_cost_right[iter]
    return (deviation<7.815)

def prunning(node):

    if not node.leaf:
        prunning(node.left)
        prunning(node.right)
        if(node.left.leaf and node.right.leaf):
            if chisquare(node):
                node.setCategory(node.categoryCount.index(max(node.categoryCount))+1)
                node.left=None
                node.right=None
                node.leaf=True





def DecisionTree(examples,features,parentexamples,node):

    """

    :param examples: Examples to learn from
    :param features: Differect features of training Date
    :param parentExamples: Example of parent
    :param node: Current Node of Tree
    :return: Decision Tree Root
    """

    if len(examples)==0:
        return( plurality(parentexamples)+1)
    #all example belongs to same class

    if isSameClass(examples):
        node.setCategory(examples[0][-1])
        node.categoryCount=categorize(examples)
        node.leaf=True
        return node
    else:
        ## split the decision tree according to highest information gain
        myentropy=entropy(examples)
            # finding out best split in data
        myinfo=infoGain(examples,myentropy)
        node.categoryCount=categorize(examples)

            # find splitvalue which has highest information gain for both features
        node.splitAttr,node.splitvalue=findHighestInfoGain(myinfo)

        left, right = split(node.splitvalue, node.splitAttr, examples)

        node1=tree()
        node1.category=(categorize(left))
        node2=tree()
        node2.category=(categorize(right))
        node.setLeft(node1)
        node.setRight(node2)

        temp=DecisionTree(left,features,examples,node1)
        if(isinstance(temp,int)):
            node.setCategory(temp)
            node.categoryCount=categorize(examples)
            return node

        temp = DecisionTree(right, features, examples, node2)
        if (isinstance(temp, int)):
            node.setCategory(temp)
            node.categoryCount = categorize(examples)
            return node

    return node

def generate_decision_boundary(instances,node):
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

    x1_corr, x2_corr = np.meshgrid(np.arange(0, 1, step), np.arange(0, 1, step))

    #print(plotting_instances)

    #now pass all these values through the classification algo

    class1_X_list, class1_Y_list, class2_X_list, class2_Y_list, class3_X_list, class3_Y_list, class4_X_list, class4_Y_list = prediction_for_plotting(plotting_instances,node)
    plt.scatter(class1_X_list,class1_Y_list, color='magenta', label='Class 1')
    plt.scatter(class2_X_list,class2_Y_list, color='orange', label='Class 2')
    plt.scatter(class3_X_list,class3_Y_list, color='purple', label='Class 3')
    plt.scatter(class4_X_list,class4_Y_list, color='gray', label='Class 4',)

    plt.scatter(X_axis_red, Y_axis_red, color = 'red', label = 'Bolts',marker='<')
    plt.scatter(X_axis_blue, Y_axis_blue, color = 'green', label = 'Nuts',marker='>')
    plt.scatter(X_axis_green, Y_axis_green, color = 'yellow', label = 'Rings',marker="*")
    plt.scatter(X_axis_yellow, Y_axis_yellow, color = 'blue', label = 'Scrap',marker='+')
    plt.xlabel('Six-fold rotational symmetry')
    plt.ylabel('Eccentricity')

    plt.title('Decision Boundary Graph')
    plt.legend()
    plt.show()


def find_accuracy(node,instances):
    '''
    finds the accuracy of the system
    :param instances: test dataset
    :param predicted_class: predicted class list
    :return:
    '''
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
        Y_predicted=[]
        predict=classify(node, each_instance)

        if each_instance[2] == 1:
            if each_instance[2] == predict:
                class1_number_of_correctly_classified+=1
                confusion_matrix[0][0]+=1
            else:
                class1_number_of_wrongly_classified+=1
                confusion_matrix[predict-1][0]+=1


        if each_instance[2] == 2:
            if each_instance[2] == predict:
                class2_number_of_correctly_classified += 1
                confusion_matrix[1][1] += 1
            else:
                class2_number_of_wrongly_classified += 1
                confusion_matrix[predict-1][1] += 1


        if each_instance[2] == 3:
            if each_instance[2] == predict:
                class3_number_of_correctly_classified += 1
                confusion_matrix[2][2] += 1
            else:
                class3_number_of_worongly_classified += 1
                confusion_matrix[predict-1][2] += 1

        if each_instance[2] == 4:
            if each_instance[2] == predict:
                class4_number_of_correctly_classified += 1
                confusion_matrix[3][3] += 1
            else:
                class4_number_of_worongly_classified += 1
                confusion_matrix[predict-1][3] += 1


    accuracy = ((class1_number_of_correctly_classified+class2_number_of_correctly_classified+class3_number_of_correctly_classified+class4_number_of_correctly_classified)/len(instances))*100
    print("The % of accuracy is:", accuracy)

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

    index=0
    for row in confusion_matrix:
        if count == 1:
            print("Predicted Class  class ",(index+1),"  ", row, "    = ",row_total[index] )
        else:
            print("                 class ",(index+1),"  ", row, "    = ",row_total[index])
        count+=1
        index+=1
    print("--------------------------------------------------")
    print("                 Total =    ", column_total)

    return accuracy, confusion_matrix



def prediction_for_plotting(plotting_instances,node):
    '''
    using this function for predicting the values for x and y values obtained from the above function (ie, generate_decision_boundary)
    :param plotting_instances:
    :return: different class list segregated using the classes and the X and Y values
    '''




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

        class_no=classify(node,each_instance)

        if class_no is 1:
            #append x1 and x2 to the class1list
            class1_X_list.append(each_instance[0])
            class1_Y_list.append(each_instance[1])

        elif class_no is 2:
            class2_X_list.append(each_instance[0])
            class2_Y_list.append(each_instance[1])

        elif class_no is 3:
            class3_X_list.append(each_instance[0])
            class3_Y_list.append(each_instance[1])

        else:
            class4_X_list.append(each_instance[0])
            class4_Y_list.append(each_instance[1])
    return class1_X_list,class1_Y_list,class2_X_list,class2_Y_list, class3_X_list,class3_Y_list, class4_X_list,class4_Y_list





def main():
    file_name=input("Enter name of data file")
    example= readCSV(file_name)
    node=tree()
    temp=DecisionTree(example,len(example[0])-1,example,node)

    print("----------------------Decision Tree----------------------")
    PrintTree(node,0)
    print()
    print()
    print("----------------------Summary----------------------------")
    maxH=temp.maxHeight(node)-1
    minH=temp.minHeight(node)-1
    nodeC=temp.numberofNode(temp)
    print("Maximum Height of Decision Tree is : ",maxH)
    print("Minum Height of Decision Tree is : ", minH)
    print("Average Height of Decision Tree is : ",temp.averageHeight(temp))
    print("Number of Nodes in Decision Tree : ",nodeC)
    print("Number of Leaf Nodes in Decision Tree : ", temp.numberofLeaf(node))
    print("Writting Decision Tree in File Decision.txt")
    with open('Decision.txt', 'wb') as output:
        writeTree(temp,output)
        output.close()
    print("Generating Decision Boundry with Training Data and Decision Tree")
    generate_decision_boundary(example,temp)

    print()
    print()
    prunning(temp)
    print("-----------------Prunned Decision Tree-------------------")
    PrintTree(temp, 0)
    print()
    print()
    print("----------------------Summary----------------------------")
    maxH = node.maxHeight(node)-1
    minH = node.minHeight(node)-1
    nodeC = temp.numberofNode(temp)
    print("Maximum Height of Prunned Decision Tree is : ", maxH)
    print("Minimum Height of Prunned Decision Tree is : ", minH)
    print("Average  Height of PrunnedDecision Tree is : ", temp.averageHeight(temp))
    print("Number of Nodes in Prunned Decision Tree : ", nodeC)
    print("Number of Leaf Prunned Nodes in Decision Tree : ", node.numberofLeaf(node))
    print("Writting Prunned Decision Tree in File Decision_Prunned.txt")
    with open('Decision_Prunned.txt', 'wb') as output:
        writeTree(temp,output)
        output.close()
    print("Generating Decision Boundry with Training Data and Prunned Decision Tree")
    generate_decision_boundary( example,temp)

if __name__ == '__main__':
    main()


