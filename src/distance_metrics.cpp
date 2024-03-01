#include <pybind11/pybind11.h>

// python_example does this https://github.com/pybind/python_example/blob/master/src/main.cpp
// I'm not exactly sure why; I'm cargo-culting it to follow the entire example
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <vector>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>


Eigen::MatrixXd distance_matrix(const Eigen::VectorXd& x){
    int n = x.size();
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            A(i, j) = std::abs(x[i] - x[j]);
            A(j, i) = A(i, j);
        }
    }
    return A;   
}

double distance_covariance(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    int n = x.size();
    Eigen::MatrixXd A = distance_matrix(x);
    Eigen::MatrixXd B = distance_matrix(y);

    double D = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            D += (A(i,j) * B(i,j))/(n * n);
        }
    }

    for (int i = 0; i < n; ++i) {
        D -= (A.row(i).sum() * B.row(i).sum())/(n * n * n);
    }

    D -= ((A.sum() * B.sum())/(n * n)) / (n * n * n * n);

    return D;
}

double distance_correlation(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    double cov_xy = distance_covariance(x, y);
    double var_x = distance_covariance(x, x);
    double var_y = distance_covariance(y, y);

    if (var_x * var_y > 0) {
    return cov_xy / std::sqrt(var_x * var_y);
    } else {
        return 0;
    }
}


Eigen::MatrixXd distance_covariance_matrix(const Eigen::MatrixXd& X) {
    int n = X.cols();
    Eigen::MatrixXd D(n, n);


    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            D(i, j) = distance_covariance(X.col(i), X.col(j));
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            D(i, j) = D(j, i);
        }
    }
    return D;
}


Eigen::MatrixXd distance_correlation_matrix(const Eigen::MatrixXd& X) {
    int n = X.cols();
    Eigen::MatrixXd D(n, n);


    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            D(i, j) = distance_correlation(X.col(i), X.col(j));
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            D(i, j) = D(j, i);
        }
    }
    return D;
}


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
