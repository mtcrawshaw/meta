import warnings

# This is to ignore warnings about tensorflow using deprecated Numpy code. This line
# must appear before importing baselines (happens in env.py).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
