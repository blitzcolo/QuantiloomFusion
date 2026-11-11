"""Registration method interface, implementations, and registration-error metrics. Depends on core, dataset, geometry, and signal."""
from .base import Registrar, registration_error
from .epipolar import EpipolarSweepRegistrar, FreeSearchRegistrar
from .params import SweepParams
from .homography import InfinityHomographyRegistrar
from .oracle import OracleRegistrar
from .resize import ResizeRegistrar
from .similarity import Similarity
