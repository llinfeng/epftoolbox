# Cursor Rules for epftoolbox

## Project Overview
Open-access library for electricity price forecasting research. Focus: reproducibility and research standards.

## Technology Stack
- Python 3.9-3.11 (TensorFlow requirement)
- TensorFlow/Keras, scikit-learn, pandas/numpy, hyperopt, statsmodels
- Jupyter Notebooks

## Code Style & Standards

### Python Code
- Follow PEP 8
- Use type hints where appropriate
- NumPy < 2.0 (pinned in setup.py)
- TensorFlow 2.x compatible

### File Naming
- snake_case for Python files/modules
- Numbered notebooks: `01_forecasting_models.ipynb`
- Use same postfix for related files (e.g., `.wav` and `.txt`)

### Project Structure
- `epftoolbox/` - main package
- `examples/` - examples
- `Notebooks/` - DS-focused examples
- `docs/` - documentation
- `datasets/` - local data (gitignored)
- `forecasts/`, `forecasts_local/` - forecast outputs

## Key Functionalities

### Models
- DNN (Deep Neural Network)
- LEAR (LASSO-regularized autoregressive)

### Metrics
- Scalar: MAE, sMAPE, MASE
- Statistical: Diebold-Mariano, Giacomini-White

### Datasets
- Five markets: EPEX-BE, EPEX-FR, EPEX-DE, NordPool, PJM
- 6 years historical prices + exogenous inputs per dataset

## Development Guidelines

### Installation
- Use `uv` for package management
- GPU: `tensorflow[and-cuda]==2.15.*`
- Pin NumPy < 2, Keras < 3 (TensorFlow 2.15 requirement)
- Use virtual environments

### Notebooks
- Self-contained, educational
- Structure: Forecasting → Evaluation → Datasets
- Call functions from main package

### Testing
- Validate GPU: `tf.config.list_physical_devices('GPU')`
- Check NumPy compatibility before changes
- Maintain reproducibility

## Code Review Checklist
- [ ] NumPy < 2.0 compatibility
- [ ] TensorFlow 2.x compatibility
- [ ] Python 3.9-3.11 compatibility
- [ ] Documentation updated (new features)
- [ ] Examples/notebooks updated (API changes)
- [ ] Reproducibility maintained

## Common Issues

### TensorFlow
- Requires Python 3.9-3.11, 64-bit
- GPU: CUDA 12.x drivers
- Validate: `tf.config.list_physical_devices('GPU')`

### NumPy
- Use `np.nan` (not `np.NaN`)
- Pin `numpy>=1,<2` in setup.py
- Avoid NumPy 2.0 API removals

### Hyperopt
- Install `setuptools<81` to quiet warnings

## Documentation
- Main: https://epftoolbox.readthedocs.io/en/latest/
- Update docs when adding features

## Citation
For significant features/modifications:
- Jesus Lago, et al. "Forecasting day-ahead electricity prices..." Applied Energy 2021; 293:116983

