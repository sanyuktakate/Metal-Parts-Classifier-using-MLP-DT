For MLP:

Run the trainMLP.py file. The file's name is already present inside the file.Do not need to take input from the user. In this the Epoch Vs SSE is plotted and
also the training set is plotted using matplotlib.
The weights file will be generated depending upon the epoch number given (0 to 10000). 

Run the executeMLP.py file. The percentage of accuracy is printed. then the confusion matrix is printed and the profit too. If the weights file, ie weights10.csv
need to be changed, then do go into the program and make the changes in the read_File() function. 




For Decision tree:


Copy trainDT.py and executeDT.py in same directory location
First Run trainDT.py python file
        you will be asked to enter data set file
        enter training data file along with extension(this file should be in same directory as programs)
        this program will save decision tree information in 2 files (Decision.txt) and(Decision_Prunned.txt) this will be input for executeDT.py

Run executeDT.py
        program will ask you to enter decision tree files which you have got fro
m running trainDT.py 
        enter that file along with extension
        program will promt to enter data file for testing
        enter data file name along with extension (this file should be in same directory as program)