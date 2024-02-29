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

double compute_mean_distance(const Eigen::VectorXd& vec) {
    double sum = 0.0;
    int n = vec.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sum += (std::abs(vec[i] - vec[j]))/(n*n);
        }
    }
    return sum;
}

double distance_covariance(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    int n = x.size();
    double mean_x = compute_mean_distance(x);
    double mean_y = compute_mean_distance(y);

    double cov = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cov += (std::abs(x[i] - x[j]) - mean_x) * (std::abs(y[i] - y[j]) - mean_y)/(n * n);
        }
    }

    return cov;
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
