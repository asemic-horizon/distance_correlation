#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>


Eigen::MatrixXd distance_matrix(const Eigen::VectorXd& x){
    // this computes a centered distance matrix
    // in the sense that it removes row and column means
    // so all rows and columns sum to 0. 

    int n = x.size();
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <n; ++j) {
            A(i, j) = std::abs(x[i] - x[j]);
        }
    }
    Eigen::VectorXd row_mean = A.rowwise().mean();
    Eigen::VectorXd col_mean = A.colwise().mean();
    // the overall matrix mean is added back because
    // we're subtracting it twice by removing row and column means
    double grand_mean = A.mean();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = A(i,j) - row_mean[i] - col_mean[j] + grand_mean;
           }
    }

    return A;   
}

double distance_covariance(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    // the direct computation of distance covariance (without 
    // clever math tricks):
    // D = expected value of (centered distance x) * (centered distance y)
    // compare Pearson's covariance formula:
    // R = expected (centered x) * (centered y)

    int n = x.size();
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    A = distance_matrix(x);
    B = distance_matrix(y);

    double D = A.cwiseProduct(B).sum() / (n*n);
    return D;
}

double distance_correlation(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    int n = x.size();
    A = distance_matrix(x);
    B = distance_matrix(y);

    double cov_xy = A.cwiseProduct(B).sum() / (n * n);
    double var_x = A.cwiseProduct(A).sum() / (n * n);
    double var_y = B.cwiseProduct(B).sum() / (n * n);
  
    return cov_xy / std::sqrt(var_x * var_y);
}


Eigen::MatrixXd distance_covariance_matrix(const Eigen::MatrixXd& X) {
    int n = X.cols();
    Eigen::MatrixXd D(n, n);

    std::vector<Eigen::MatrixXd> As(n);

    #pragma omp parallel for
    for (int i = 0; i<n; ++i) {
        As[i] = distance_matrix(X.col(i));
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            D(i,j) = As[i].cwiseProduct(As[j]).sum() / (n * n);
        }
    }
    return D;
}


Eigen::MatrixXd distance_correlation_matrix(const Eigen::MatrixXd& X) {
    int n = X.cols();
    Eigen::MatrixXd D(n, n);

    std::vector<Eigen::MatrixXd> As(n);
    std::vector<double> var(n);
    double cov_xy;

    #pragma omp parallel for
    for (int i = 0; i<n; ++i) {
        As[i] = distance_matrix(X.col(i));
        var[i] = As[i].cwiseProduct(As[i]).sum() / (n * n);
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cov_xy = As[i].cwiseProduct(As[j]).sum() / (n * n);
            D(i,j) = cov_xy / std::sqrt(var[i] * var[j]);
        }
    }

    return D;
}

// these macros so that python's setup.py can inject a version number here
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(distance_metrics, m) {
    m.def("distance_covariance", &distance_covariance, "Compute distance covariance between two vectors");
    m.def("distance_correlation", &distance_correlation, "Compute distance correlation between two vectors");
    m.def("distance_covariance_matrix", &distance_covariance_matrix, "Compute distance covariance matrix");
    m.def("distance_correlation_matrix", &distance_correlation_matrix, "Compute distance correlation matrix");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
