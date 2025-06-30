from .mimc import MIMCRun, MIMCItrData
from . import plot
from .db import MIMCDatabase
from . import miproj

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("mimclib")
except PackageNotFoundError:
    __version__ = "Not installed"
