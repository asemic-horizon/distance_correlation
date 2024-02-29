# distance correlation

Distance correlation is a measure of dependence between two vectors. It's particularly known for the nice property that two uncorrelated vectors are guaranteed to be independent.

In slightly more detail: given X and Y random vectors, distance covariance is a metric (called the [energy distance](https://en.wikipedia.org/wiki/Energy_distance)) of how distant is the actual joint distribution (X,Y) from the alternate distribution that would hold if X and Y were independent. [Wikipedia has the technical details](https://en.wikipedia.org/wiki/Distance_correlation), but to illustrate its main properties we'll borrow the following chart, with depicts a few point cloud-type datasets and their distance correlation:

![Distance Correlation - examples](https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Distance_Correlation_Examples.svg/1024px-Distance_Correlation_Examples.svg.png)

- Unlike Pearson correlation, distance correlation is bounded between 0 and 1, and doesn't have an interpretation of linear directionality, Therefore negatively sloped lines have perfect correlation = 1.
- Distance correlation is quite adept at detecting nonlinear structures. The Pearson correlation for all the charts in the third row is 0, 

I've always been interested in distance correlation and tried to use it in my datasets, but it's quite computation-intensive in ways that aren't easy to vectorize for numpy optimizations. This package relies on a custom-written C++ extension that leverages [pybind11](https://github.com/pybind/pybind11) for Python integration and the excellent [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) library for C++ linear algebra that saves the heartache of manual memory management.
