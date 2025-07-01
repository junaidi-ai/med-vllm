"""
Path-related utility functions for configuration handling.

This module provides functions for working with file system paths in the
context of configuration management, including path resolution and validation.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

PathLike = Union[str, os.PathLike, Path]


def ensure_path(path: PathLike, must_exist: bool = False) -> Path:
    """Convert a path-like object to a Path and optionally check for existence.

    Args:
        path: The path to convert (str, Path, or os.PathLike)
        must_exist: If True, raises FileNotFoundError if path doesn't exist

    Returns:
        Path: The converted Path object

    Raises:
        FileNotFoundError: If must_exist is True and path doesn't exist
        TypeError: If path is not a valid path-like object
    """
    if isinstance(path, Path):
        path_obj = path
    elif isinstance(path, (str, os.PathLike)):
        path_obj = Path(path).expanduser().absolute()
    else:
        msg = f"Expected path-like object, got {type(path).__name__}"
        raise TypeError(msg)

    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path_obj}")

    return path_obj


def resolve_config_path(
    path: PathLike,
    search_paths: Optional[List[PathLike]] = None,
    file_name: Optional[str] = None,
) -> Path:
    """Resolve a configuration file path with optional search paths.

    Args:
        path: The path to resolve (can be relative or absolute)
        search_paths: Optional list of directories to search in
        file_name: Optional filename to append if path is a directory

    Returns:
        Path: The resolved absolute path

    Raises:
        FileNotFoundError: If the path cannot be resolved
    """
    path = ensure_path(path)

    # If path exists and is a file, return it
    if path.is_file():
        return path.absolute()

    # If path is a directory and filename is provided, try that
    if path.is_dir() and file_name:
        file_path = path / file_name
        if file_path.is_file():
            return file_path.absolute()

    # Try searching in additional paths
    if search_paths:
        for search_path in search_paths:
            search_path = ensure_path(search_path)
            if not search_path.is_dir():
                continue

            # Try the path directly
            if (search_path / path).is_file():
                return (search_path / path).absolute()

            # Try with filename if provided
            if file_name and (search_path / file_name).is_file():
                return (search_path / file_name).absolute()

    raise FileNotFoundError(f"Could not resolve config path: {path}")


def find_config_file(
    file_name: str,
    search_dirs: Optional[List[PathLike]] = None,
    extensions: Optional[List[str]] = None,
) -> Optional[Path]:
    """Find a configuration file in standard locations.

    Args:
        file_name: Base name of the file to find (without extension)
        search_dirs: Optional list of directories to search in
        extensions: Optional list of file extensions to try (including .)

    Returns:
        Path to the found file, or None if not found
    """
    if not search_dirs:
        search_dirs = [
            Path.cwd(),
            Path.home() / ".config" / "med-vllm",
            Path("/etc/med-vllm"),
        ]

    if not extensions:
        extensions = [".yaml", ".yml", ".json", ""]

    for search_dir in search_dirs:
        search_dir = ensure_path(search_dir)
        if not search_dir.is_dir():
            continue

        for ext in extensions:
            file_path = search_dir / f"{file_name}{ext}"
            if file_path.is_file():
                return file_path.absolute()

    return None


def get_relative_path(path: Path, base: Optional[Path] = None) -> str:
    """Get a path relative to a base directory, or make it absolute.

    Args:
        path: The path to make relative
        base: The base directory (defaults to current working directory)

    Returns:
        str: The relative path if possible, otherwise absolute path as string
    """
    if base is None:
        base = Path.cwd()

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path.absolute())
