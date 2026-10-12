"""Registration method interface, implementations, and registration-error metrics. Depends on core, dataset, geometry, and signal."""
from .base import Registrar, registration_error
from .homography import InfinityHomographyRegistrar
from .oracle import OracleRegistrar
from .resize import ResizeRegistrar
