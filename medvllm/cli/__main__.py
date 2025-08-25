"""Module entry point for `python -m medvllm.cli`.

This delegates to the Click CLI defined in `medvllm.cli.__init__`.
"""

from . import cli  # noqa: F401

if __name__ == "__main__":
    cli()  # type: ignore[no-untyped-call]
