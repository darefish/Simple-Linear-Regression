#include <iostream>
#include <fstream>
#include <math.h>

#include <Eigen/Dense>

#include "utils.cpp"

using namespace std;
using Eigen::MatrixXd;

int main(int argc, char** argv){

  if(argc < 3){
    std::cout << "Please enter the requires parameters :\n1) Training Data File 2)Learning Rate 3)Iterations\n" ;
    return 1;
  }
  
  std::string filepath = argv[1];
  double learning_rate = std::stod( argv[2] );
  double iterations =  std::stod( argv[3] );
  
  // Fetching the data from the csv file 
  std::cout << std::endl << " Reading the file " << filepath;
  MatrixXd data = load_csv<MatrixXd>(filepath);
  std::cout << std::endl << " Number of traning examples = " << data.rows() << std::endl;
  
  std::cout << "Starting gradient descent optimization : ";
  RowVector2d parameters = trainLinearRegressionModel(data, iterations, learning_rate);
  std::cout << std::endl << " Slope  = " << parameters(0) << std::endl << " Intercept = " << parameters(1) << std::endl;
  
  ofstream output;
  output.open("result.txt");
  output << " Slope = " << parameters(0) << "\n Intercept = " << parameters(1);  
  output.close();
  
  return 0;
}