from .distance_metrics import distance_covariance as _distance_covariance
from .distance_metrics import distance_correlation as _distance_correlation
from .distance_metrics import distance_covariance_matrix as _distance_covariance_matrix
from .distance_metrics import distance_correlation_matrix as _distance_correlation_matrix
from numpy import ndarray

def distance_covariance(x: ndarray, y: ndarray) -> float:
    """Compute distance covariance between two vectors with the same length/dimension.

    Args:
        x (ndarray): vector (1-dimensional ndarray)
        y (ndarray): vector (1-dimensional ndarray)

    Returns:
        float: Distance covariance between x and y
    """
    return _distance_covariance(x, y)



def distance_correlation(x: ndarray, y: ndarray) -> float:
    """Compute distance correlation coefficient between two vectors with the same length/dimension.

    Args:
        x (ndarray): vector (1-dimensional ndarray)
        y (ndarray): vector (1-dimensional ndarray)

    Returns:
        float: Distance correlation between x and y
    """
    return _distance_correlation(x, y)

def distance_covariance_matrix(X: ndarray) -> ndarray:
    """Compute a distance covariance matrix for two data matrices,
    so the (i,j)-th element is distance_covariance(X[i], Y[j]).

    Args:
        X (ndarray): data matrix (2-dimensional ndarray)
        Y (ndarray): data matrix (2-dimensional ndarray)

    Returns:
        ndarray: Distance covariance matrix between X and Y
    """
    return _distance_covariance_matrix(X)

def distance_correlation_matrix(X: ndarray) -> ndarray:
    """Compute a distance correlation matrix for two data matrices,
    so the (i,j)-th element is distance_correlation(X[i], Y[j]).

    Args:
        X (ndarray): data matrix (2-dimensional ndarray)
        Y (ndarray): data matrix (2-dimensional ndarray)

    Returns:
        ndarray: Distance correlation matrix between X and Y
    """
    return _distance_correlation_matrix(X)
