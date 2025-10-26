"""Compatibility layer for the legacy :mod:`lensing_ssc.core.base` package.

The foundational components now live under :mod:`lensing_ssc.core.foundation`.
Importing from :mod:`lensing_ssc.core.base` will continue to work, but a
:class:`DeprecationWarning` is emitted to help consumers migrate to the new
package structure.
"""

from warnings import warn

from ..foundation import *  # noqa: F401,F403
from ..foundation import __all__ as _foundation_all

warn(
    "'lensing_ssc.core.base' is deprecated; use 'lensing_ssc.core.foundation' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_foundation_all)
