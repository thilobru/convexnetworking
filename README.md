# convexNetworking

This repository contains code for a machine learning-based approach to the convex gating algorithm, designed for gating clusters of observations, particularly in the context of flow cytometry data analysis.

### Project Structure

The main components of the project are located within the `convexgating` subfolder. The `convexnetworking` subfolder appears to contain related experimental work, exploring neural network-based approaches.

* `convexgating/`:

  * `__init__.py`: Initializes the `convexgating` package, importing functions from the other modules.

  * `helper.py`: Contains various helper functions for data preprocessing, marker selection heuristics (based on heuristic, decision tree, or linear SVM), gradient descent optimization for finding gate locations, and geometric calculations related to convex hulls and hyperplanes.

  * `hyperparameters.py`: Defines model hyperparameters and visualization settings for the gating strategy.

  * `plotting.py`: Includes functions for visualizing the gating strategy and its performance using scatter plots, heatmaps, and metrics plots.

  * `simulation.py`: Provides functions for simulating FACS data, either as a complete dataset or per population.

  * `tools.py`: Contains the core functions for learning the gating strategy, including `gating_strategy` and `FIND_GATING_STRATEGY`, which utilize adaptive grid search and 2D gate finding based on the helper functions.

  * `cg_toy_example.ipynb`: A tutorial notebook demonstrating how to use the `convexgating` library with a toy dataset, covering data generation, clustering (using `louvain`), and applying the `gating_strategy` function.

  * `cg_toy_example_advanced.ipynb`: An advanced tutorial notebook showing how to access more detailed results from the gating algorithm, such as gate locations and cell membership.

* `convexnetworking/`:

  * `ConvexNetwork.ipynb`, `networkExperiments.ipynb`, `wrongConvexNetwork.ipynb`: Jupyter notebooks exploring neural network models and experiments, potentially for learning convex hulls or related concepts.

  * `convexnetworktools.py`: Contains some helper functions, including a `classify_points` function, likely used in the neural network experiments.

* `convexNetworking.ipynb`: A top-level notebook that loads some packages and generates toy data, but the main gating function appears incomplete.

* `Test.ipynb`: A notebook with TensorFlow code, including a custom `in_convex_hull` function (not fully implemented) and experiments with a simple Keras Sequential model and loss functions like 'BinaryCrossentropy' and 'Hinge'.

* `test.py`: A simple Python script with a function `lettersum`.

### Getting Started

#### Prerequisites

The core `convexgating` library depends on several Python packages. Based on the import statements in the provided files, you will need:

* `convexgating` (the library itself, included in this repository)

* `scanpy`

* `anndata`

* `pandas`

* `numpy`

* `sklearn` (from `sklearn.datasets` and `sklearn.metrics`)

* `torch` (for gradient descent in `helper.py`)

* `matplotlib` (for plotting)

* `tensorflow` and `keras` (for the experimental neural network code)

* `louvain` (for clustering in the toy example)

You can install most of these using pip:

```text
pip install scanpy anndata pandas numpy scikit-learn torch matplotlib louvain tensorflow keras
```

Ensure the `convexgating` package is accessible in your Python environment. This might involve adding the `convexNetworking/convexgating` directory to your Python path or installing it as a local package.

#### Running the Examples

Explore the tutorial notebooks in the `convexgating` folder to understand how to use the library:

* `convexgating/cg_toy_example.ipynb`: Provides a basic walkthrough of applying the gating strategy.

* `convexgating/cg_toy_example_advanced.ipynb`: Shows how to access more detailed results from the gating algorithm.

### Overview

The `convexgating` library implements a method to find a series of optimal gates (defined by hyperplanes) to isolate specific cell clusters in multi-dimensional data. This is achieved through an adaptive grid search combined with a gradient descent optimization process to determine the best marker combinations and gate locations. The approach incorporates heuristics and can optionally use PCA for gate initialization. Performance metrics and visualizations are available to evaluate the effectiveness of the learned gating strategy.

The experimental code in the `convexnetworking` folder appears to be a separate investigation into using neural networks to potentially learn aspects of convex shapes or gating.

Feel free to explore the code and the examples to understand the implementation details and apply the convex gating strategy to your own data.