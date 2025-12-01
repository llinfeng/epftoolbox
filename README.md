# epftoolbox

The epftoolbox is the first open-access library for driving research in electricity price forecasting. Its main goal is to make available a set of tools that ensure reproducibility and establish research standards in electricity price forecasting research.

The library has been developed as part of the following article:

- Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". *Applied Energy* 2021; 293:116983. [https://doi.org/10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983).

The library is distributed under the AGPL-3.0 License and it is built on top of scikit-learn, tensorflow, keras, hyperopt, statsmodels, numpy, and pandas. 

Website: [https://epftoolbox.readthedocs.io/en/latest/](https://epftoolbox.readthedocs.io/en/latest/)

## Getting started
Download the repository and navigate into the folder
```bash
$ git clone https://github.com/jeslago/epftoolbox.git
$ cd epftoolbox
```
[Optional] Create an environment to avoid conflicts, e.g. using [conda](https://docs.conda.io/en/latest/):
```bash
conda create --name epftoolbox python=3.10
conda activate epftoolbox
```

Install using pip
```bash
$ pip install .
```
Navigate to the examples folder and check the existing examples to get you started. The examples include several applications of the two state-of-the art forecasting model: a deep neural net and the LEAR model.

### Troubleshooting
The installation will fail if [tensorflow](https://www.tensorflow.org/install/pip#macos) requirements are not met. As of 2023/11/13, they require a python version between 3.9 and 3.11 and a 64bits python version.

NumPy compatibility: the code uses `np.nan` and we pin `numpy>=1,<2` in `setup.py` to match TensorFlow 2.x and avoid NumPy 2.0 API removals (e.g., `np.NaN`). If you override the pin, expect to stay on NumPy 1.x until TensorFlow officially supports 2.x.

For any other problem, open an [issue](https://github.com/jeslago/epftoolbox/issues).

## Documentation
The documentation can be found [here](https://epftoolbox.readthedocs.io/en/latest/). It provides an introduction to the library features and explains all functionalities in detail. Note that the documentation is still being built and some functionalities are still undocumented.

## Features
The library provides easy access to a set of tools and benchmarks that can be used to evaluate and compare new methods for electricity price forecasting.

### Forecasting models
The library includes two state-of-the-art forecasting models that can be automatically employed in any day-ahead market without the need of expert knowledge. At the moment, the library comprises two main models:
  * One based on a deep neural network
  * A second based on an autoregressive model with LASSO regulazariton (LEAR). 

### Evaluation metrics
Standard evaluation metrics for electricity price forecasting including:
* Multiple scalar metrics like MAE, sMAPE, or MASE.
* Two statistical tests (Diebold-Mariano and Giacomini-White) to evaluate statistical differents in forecasting performance.

### Day-ahead market datasets
Easy access to five datasets comprising 6 years of data each and representing five different day-ahead electricity markets: 
* The datasets represents the EPEX-BE, EPEX-FR, EPEX-DE, NordPool, and PJM markets. 
* Each dataset contains historical prices plus two time series representing exogenous inputs.

### Available forecasts
Readily available forecasts of the state-of-the-art methods so that researchers can evaluate new methods without re-estimating the models.


## Citation
If you use the epftoolbox in a scientific publication, we would appreciate citations to the following paper:

- Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". *Applied Energy* 2021; 293:116983. [https://doi.org/10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983).


Bibtex entry::

    @article{epftoolbox,
    title = {Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark},
    journal = {Applied Energy},
    volume = {293},
    pages = {116983},
    year = {2021},
    doi = {https://doi.org/10.1016/j.apenergy.2021.116983},
    author = {Jesus Lago and Grzegorz Marcjasz and Bart {De Schutter} and Rafał Weron}
    }
    
    
----

# Notebooks for DS
If you want a fresh environment via `uv` (fast, offline-friendly installer) and GPU support:
1) `uv venv .venv && source .venv/bin/activate`
2) `uv pip install --upgrade pip`
3) `uv pip install "setuptools<81"`  (quiet the hyperopt/pkg_resources warning)
4) GPU-enabled TensorFlow by default:  
   `uv pip install "tensorflow[and-cuda]==2.15.*" -e .`  
   - Uses CUDA/cuDNN wheels (driver must support CUDA 12.x).  
   - Pins NumPy <2 and Keras <3 via `setup.py` to match TF 2.15.  
   - Validate GPU visibility:  
     `python - <<'PY'\nimport tensorflow as tf\nprint(tf.config.list_physical_devices('GPU'))\nPY`
   - If you keep seeing “Could not find cuda drivers”: verify `nvidia-smi`, install a matching driver, and ensure the venv uses this install.

The `Notebooks/` folder is added to call functions defined in this repo. The
goal is to spoonfeed the users with working examples in Jupyter notebook format,
so that one native to the DS field can follow.

Noetbooks are created with help of Codex, and the ordering of notebooks is
arranged so that it matches the narratives on the feature in this repo, namely:
Forecatsing, Evaluation and dataset of DA markets and predicted values in CSV
format.
