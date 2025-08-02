"""
OpenGait main package
"""

# Import key submodules for easier access
try:
    from . import modeling
    from . import data  
    from . import utils
    from . import evaluation
    
    __all__ = ['modeling', 'data', 'utils', 'evaluation']
    
except ImportError as e:
    # Graceful fallback if some modules aren't available
    import warnings
    warnings.warn(f"Some OpenGait modules couldn't be imported: {e}")
    __all__ = []

# Set up numpy printing for consistent output across the package
try:
    from numpy import set_printoptions
    set_printoptions(suppress=True, formatter={'float': '{:0.2f}'.format})
except ImportError:
    pass