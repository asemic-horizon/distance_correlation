from .distance_metrics import distance_covariance as _distance_covariance
from .distance_metrics import distance_correlation as _distance_correlation
from .distance_metrics import distance_covariance_matrix as _distance_covariance_matrix
from .distance_metrics import distance_correlation_matrix as _distance_correlation_matrix
from numpy import ndarray

def distance_covariance(x: ndarray, y: ndarray) -> float:
    """Compute distance covariance between two vectors with the same length/dimension.

    Let x, y be samples of random variables X, Y. Consider the following:
    - The actual joint distribution (X,Y)
    - The alternative joint distribution that we would have if X and Y were independent

    Distance covariance is the energy distance between these two cases. Therefore, it
    measures how far from being independent X and Y are.

    Like Pearson covariance, distance covariance is not normalized and grows in scale
    as the variables X, Y grow in scale, which can make it hard to interpret. Consider
    using distance_correlation instead.

    Args:
        x (ndarray): vector (1-dimensional ndarray)
        y (ndarray): vector (1-dimensional ndarray)

    Returns:
        float: Distance covariance between x and y
    """
    return _distance_covariance(x, y)


def distance_variance(x: ndarray) -> float:
    """Compute the distance variance of a vector.
    This is equivalent to the covariance of a vector with itself.

    Args:
        x (ndarray): vector (1-dimensional ndarray)

    Returns:
        float: Distance variance of x
    """
    return _distance_covariance(x, x)

def distance_correlation(x: ndarray, y: ndarray) -> float:
    """Compute distance correlation coefficient between two vectors with the same length/dimension.
    This is defined by analogy to the Pearson correlation coefficient, i.e.

    cor(X, Y) = cov(X, Y) / sqrt(cov(X, X) * cov(Y, Y))

    This definition gives some notable properties:

    - This coefficient is bounded between 0 and 1: even negative relationships have a positive correlation.
    - If X, Y are independent random variables, then cor(X, Y) = 0 (this isn't true of correlation metrics
    in general.
    - If cor(X, Y) = 1, then Y = aX + b for some a, b (i.e. Y is a linear transformation of X).

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

     Distance covariances can be large, which gives rise to numerical stability issues in a
     context where different columns have different scale orders of magnitude. While care was
     taken to stabilize the computation of distance covariances in the C++ implementation, I'm not sure
     passing the resulting matrices around in Python is a good idea. Consider using distance_correlation
     or standardizing your variables beforehand.


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
