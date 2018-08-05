# Simple Linear Regression 

Implementing uni-variate linear regression using the Eigen library in C++ 

## Building the project 
#### The Eigen Library can be used directly as a set of header files there is no need for linking.
$ g++ -I./lib/ ./src/main.cpp -o ./bin/LinearRegression

## Running the project
#### Command line parameters for running the program : 
1. The path of the csv file containing the data 
2. The learning rate for the gradient descent optimization routine 
3. The iterations for the optimization routine 

$ ./bin/LinearRegression ./data/train.csv 0.00005 50

#### The program returns the learned slope and intercept values in a file named "results.txt" in the same directory. 


