# based on https://github.com/pybind/python_example/
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext

from setuptools import setup, find_packages
import os, urllib.request, zipfile
__version__ = "0.0.2"

with open('README.md', 'r') as f:
    long_description = f.read()

# this class is ChatGPT's idea of how to bring eigen3 in. I'm sure there's a more elegant way to do this.
class CustomBuildExt(build_ext):
    def run(self):
        eigen3_dir = self.download_and_extract_eigen3()
        for ext in self.extensions:
            ext.include_dirs.append(eigen3_dir)  # Add Eigen3 include directory
        super().run()

    def download_and_extract_eigen3(self):
        eigen3_version = "3.4.0"
        eigen3_url = f"https://gitlab.com/libeigen/eigen/-/archive/{eigen3_version}/eigen-{eigen3_version}.zip"
        temp_dir = os.path.join(os.getcwd(), 'temp_eigen3')
        zip_path = os.path.join(temp_dir, 'eigen3.zip')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download Eigen3 zip
        print(f"Downloading Eigen3 from {eigen3_url}...")
        urllib.request.urlretrieve(eigen3_url, zip_path)
        
        # Extract the zip file
        print(f"Extracting Eigen3 to {temp_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Return the path to the Eigen3 include directory
        eigen3_include_dir = os.path.join(temp_dir, f'eigen-{eigen3_version}')
        return eigen3_include_dir


ext_modules = [
    Pybind11Extension(
        "distance_correlation.distance_metrics",
        ["src/distance_correlation/distance_metrics.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=['-std=c++11', '-O3', '-fopenmp'],  
        extra_link_args=['-fopenmp']
    ),
]

setup(
    name="distance_correlation",
    version=__version__,
    author="Diego Navarro",
    author_email="the.electric.me@gmail.com",
    url="https://github.com/asemic-horizon/distance_correlation",
    description="Computes distance covariances/correlations between vectors and distance covariance/correlation matrices for data matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",  
    ext_modules=ext_modules,
    #extras_require={"test": "pytest"}, no test suite yet
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.9",
    packages=['distance_correlation'],
    package_dir={"distance_correlation": "src/distance_correlation/"},
)