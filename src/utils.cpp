#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>

using namespace Eigen;

template<typename M>


/**
    Reads in the data from a csv file.

    @param Path of the csv file.
    @return A Eigen map containing the data.
*/
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
          if (!cell.empty()){
            values.push_back(std::stod(cell));
          }
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

/**
    Trains a linear regression model.
    Loss : Least Squared Error , Optimizer : Vanilla Gradient Descent

    @param A eigen matrix object containing the training data.
    @return The linear regression model parameters learned. 
            The first element of the parameter vector is the slope. 
            The second element of the parameter vector is the intercept.
*/
RowVector2d trainLinearRegressionModel(MatrixXd data, int iterations, double learning_rate){
    //Extracting the target vector
    MatrixXd Y = data.rightCols(1);
    int m = Y.rows();
    
    //Adding a column of ones in the feature matrix
    MatrixXd X = data.leftCols(1);
    RowVectorXd ones;
    ones = RowVectorXd::Constant( X.rows(), 1.0 ) ;  
    X.conservativeResize( X.rows(), X.cols()+1 );
    X.col( X.cols()-1 ) = ones;

    // Initializing the slope and intercept parameters 
    RowVector2d parameters; 
    parameters << 0, 0;

    // Running the gradient descent optimization routine
    MatrixXd H;
    double loss; 
    std::cout << std::endl << " Running gradient descent for "<< iterations << " iterations " << std::endl;
    for ( int i=0; i< iterations; i++){
        H = ( X * parameters.transpose() );
        //Updating parameters
        parameters(0) = parameters(0) - ( learning_rate / m * ( H - Y ).cwiseProduct( X.leftCols(1)).array().sum() )  ;
        parameters(1) = parameters(1) - ( learning_rate / m * ( H - Y ).array().sum()  )  ;
        H = ( X * parameters.transpose() );
        loss = (H-Y).array().square().sum() / 2 / m;
        std::cout << std::endl << " Loss at iteration " << i << " = " << loss ;
    }
    std::cout << std::endl << " Linear Regression Model tranied " << std::endl;
    
    return parameters;
}


