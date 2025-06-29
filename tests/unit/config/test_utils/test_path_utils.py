"""
Tests for path utility functions.
"""

import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Import the actual implementation
from medvllm.medical.config.utils.path_utils import (
    ensure_path,
    resolve_config_path,
    find_config_file,
    get_relative_path
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
    
    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_resolve_config_path_file(
        self, mock_is_file: MagicMock, mock_exists: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """Test resolve_config_path with existing file."""
        # Given
        mock_resolve.return_value = Path("/resolved/path")
        
        # When
        result = resolve_config_path("test.yaml")
        
        # Then
        assert result == Path("/resolved/path")
        mock_resolve.assert_called_once()
    
    @patch("pathlib.Path.exists", return_value=False)
    def test_resolve_config_path_not_found(self, mock_exists: MagicMock) -> None:
        """Test resolve_config_path with non-existent file."""
        with pytest.raises(FileNotFoundError):
            resolve_config_path("nonexistent.yaml")
    
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.is_file", return_value=False)
    @patch("pathlib.Path.exists", return_value=True)
    def test_find_config_file_directory(
        self, mock_exists: MagicMock, mock_is_file: MagicMock, mock_glob: MagicMock
    ) -> None:
        """Test find_config_file with directory input."""
        # Given
        mock_glob.return_value = [
            Path("/test/dir/config.yaml"),
            Path("/test/dir/config.json"),
        ]
        
        # When
        result = find_config_file("/test/dir")
        
        # Then
        assert result == Path("/test/dir/config.yaml")  # Should return first match
        mock_glob.assert_called_once_with("*.[jy]a?ml")
    
    @patch("pathlib.Path.glob", return_value=[])
    @patch("pathlib.Path.is_file", return_value=False)
    @patch("pathlib.Path.exists", return_value=True)
    def test_find_config_file_not_found(
        self, mock_exists: MagicMock, mock_is_file: MagicMock, mock_glob: MagicMock
    ) -> None:
        """Test find_config_file with no matching files."""
        with pytest.raises(FileNotFoundError):
            find_config_file("/test/dir")
    
    def test_get_relative_path(self) -> None:
        """Test getting a relative path."""
        # Given
        base = "/base/directory"
        path = "/base/directory/subdir/file.txt"
        
        # When
        result = get_relative_path(path, base)
        
        # Then
        assert result == Path("subdir/file.txt")
    
    def test_get_relative_path_outside_base(self) -> None:
        """Test getting a relative path outside base directory."""
        # Given
        base = "/base/directory"
        path = "/other/directory/file.txt"
        
        # When/Then
        with pytest.raises(ValueError, match="not inside base directory"):
            get_relative_path(path, base)
    
    @patch("os.path.commonpath")
    @patch("os.path.abspath")
    def test_get_relative_path_platform_specific(
        self, mock_abspath: MagicMock, mock_commonpath: MagicMock
    ) -> None:
        """Test platform-specific path handling."""
        # Given
        mock_abspath.side_effect = lambda x: x  # Return as-is
        mock_commonpath.return_value = "/common"
        
        # When/Then - Should handle both forward and backslashes
        for path in ["C:\\path\\file.txt", "/unix/path/file.txt"]:
            get_relative_path(path, "/common")
            mock_commonpath.assert_called()
