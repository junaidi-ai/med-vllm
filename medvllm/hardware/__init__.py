"""Hardware detection and compatibility helpers for Med vLLM.

This subpackage provides:
- detect: utilities to inspect available devices/backends on the host
- compat: thin shims to gate optional accelerators without hard deps
"""

from .detect import get_hardware_profile, list_available_backends  # noqa: F401
