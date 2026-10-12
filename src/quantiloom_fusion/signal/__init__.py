"""Explicit normalisation from product signals to fusion inputs. Depends on core and dataset."""
from .colour import demosaic_bilinear, luminance, rgb_to_ycbcr, ycbcr_to_rgb
from .normalize import Normalizer
