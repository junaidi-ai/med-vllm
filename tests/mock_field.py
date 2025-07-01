"""Mock for dataclasses.field() to handle description parameter."""

import sys
from dataclasses import field as dataclass_field


def field(*args, **kwargs):
    """Mock field function that removes description parameter."""
    if "description" in kwargs:
        del kwargs["description"]
    return dataclass_field(*args, **kwargs)


# Replace the field function in the dataclasses module
import dataclasses  # noqa: E402

if hasattr(dataclasses, "field"):
    sys.modules["dataclasses"].field = field
