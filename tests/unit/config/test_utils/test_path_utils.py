"""
Tests for path utility functions.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the actual implementation
from medvllm.medical.config.utils.path_utils import (
    ensure_path,
    find_config_file,
    get_relative_path,
    resolve_config_path,
)


class TestPathUtils:
    """Test cases for path utility functions."""

    def test_ensure_path_string(self) -> None:
        """Test ensure_path with string input."""
        # When
        result = ensure_path("/test/path")

        # Then
        assert isinstance(result, Path)
        assert str(result) == "/test/path"

    def test_ensure_path_path(self) -> None:
        """Test ensure_path with Path input."""
        # Given
        path = Path("/test/path")

        # When
        result = ensure_path(path)

        # Then
        assert result is path  # Should return same object

    def test_ensure_path_invalid(self) -> None:
        """Test ensure_path with invalid input."""
        with pytest.raises(TypeError):
            ensure_path(123)  # type: ignore

    def test_resolve_config_path_file(self, tmp_path: Path) -> None:
        """Test resolve_config_path with existing file."""
        # Create a test file
        test_file = tmp_path / "test.yaml"
        test_file.write_text("test: data")

        # When
        result = resolve_config_path(test_file)

        # Then
        assert result == test_file.absolute()
        assert result.exists()
        assert result.is_file()

    @patch("pathlib.Path.exists", return_value=False)
    def test_resolve_config_path_not_found(self, mock_exists: MagicMock) -> None:
        """Test resolve_config_path with non-existent file."""
        with pytest.raises(FileNotFoundError):
            resolve_config_path("nonexistent.yaml")

    def test_find_config_file_directory(self, tmp_path: Path) -> None:
        """Test find_config_file with directory input."""
        # Create a test config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("test: config")

        # Test with the temporary directory
        result = find_config_file("test_config", search_dirs=[tmp_path])

        # Verify the file was found
        assert result is not None
        assert result.name == "test_config.yaml"
        assert result.exists()

    @patch("pathlib.Path.is_file", return_value=False)
    @patch("pathlib.Path.is_dir", return_value=True)
    def test_find_config_file_not_found(
        self, mock_is_dir: MagicMock, mock_is_file: MagicMock
    ) -> None:
        """Test find_config_file with no matching files."""
        with patch("pathlib.Path") as mock_path:
            # Configure the mock to simulate no files found
            mock_path.return_value = MagicMock(is_file=MagicMock(return_value=False))

            # When/Then
            result = find_config_file("nonexistent", search_dirs=[Path("/test/dir")])
            assert result is None

    @patch("pathlib.Path.cwd")
    def test_get_relative_path(self, mock_cwd: MagicMock) -> None:
        """Test getting a relative path."""
        # Setup mock for current working directory
        mock_cwd.return_value = Path("/base")

        # When path is relative to base
        path = Path("/base/dir/file.txt")
        result = get_relative_path(path)

        # Then should return relative path
        assert result == "dir/file.txt"

        # When path is not relative to base
        path = Path("/other/dir/file.txt")
        result = get_relative_path(path)

        # Then should return absolute path as string
        assert result == "/other/dir/file.txt"

    @patch("pathlib.Path.cwd")
    def test_get_relative_path_outside_base(self, mock_cwd: MagicMock) -> None:
        """Test getting a relative path outside base directory."""
        # Setup mock for current working directory
        mock_cwd.return_value = Path("/base/directory")

        # When path is outside the base directory
        path = Path("/other/directory/file.txt")
        result = get_relative_path(path)

        # Then should return absolute path as string
        assert result == "/other/directory/file.txt"

    @patch("pathlib.Path.cwd")
    def test_get_relative_path_platform_specific(self, mock_cwd: MagicMock) -> None:
        """Test platform-specific path handling."""
        # Setup mock for current working directory
        mock_cwd.return_value = Path("/common")

        # Test with a path that's relative to the base
        path = Path("/common/subdir/file.txt")
        result = get_relative_path(path)
        assert result == "subdir/file.txt"

        # Test with a path that's not relative to the base
        path = Path("/other/dir/file.txt")
        result = get_relative_path(path)
        assert result == "/other/dir/file.txt"
