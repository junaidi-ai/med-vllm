"""
Tests for type utility functions.
"""

from typing import Any, Dict, List, Optional, Union

import pytest

# Import the actual implementation
from medvllm.medical.config.utils.type_utils import (
    convert_string_to_type,
    get_dict_types,
    get_list_item_type,
    get_optional_type,
    get_union_types,
    is_basic_type,
    is_dict_type,
    is_list_type,
    is_optional_type,
    is_union_type,
    validate_type,
)


class TestTypeUtils:
    """Test cases for type utility functions."""

    # Test is_optional_type
    @pytest.mark.parametrize(
        "type_hint,expected",
        [
            (Optional[int], True),
            (Optional[str], True),
            (Union[None, int], True),
            (Union[int, None], True),
            (int, False),
            (str, False),
            (List[int], False),
            (Dict[str, int], False),
        ],
    )
    def test_is_optional_type(self, type_hint: Any, expected: bool) -> None:
        """Test is_optional_type function."""
        assert is_optional_type(type_hint) == expected

    # Test get_optional_type
    @pytest.mark.parametrize(
        "type_hint,expected",
        [
            (Optional[int], int),
            (Optional[str], str),
            (Union[None, int], int),
            (Union[int, None], int),
        ],
    )
    def test_get_optional_type(self, type_hint: Any, expected: type) -> None:
        """Test get_optional_type function."""
        assert get_optional_type(type_hint) == expected

    # Test is_union_type
    @pytest.mark.parametrize(
        "type_hint,expected",
        [
            (Union[int, str], True),
            (Union[bool, int, str], True),
            (int, False),
            (Optional[int], False),  # Optional is handled separately
            (List[int], False),
        ],
    )
    def test_is_union_type(self, type_hint: Any, expected: bool) -> None:
        """Test is_union_type function."""
        assert is_union_type(type_hint) == expected

    # Test get_union_types
    @pytest.mark.parametrize(
        "type_hint,expected",
        [
            (Union[int, str], {int, str}),
            (Union[bool, int, str], {bool, int, str}),
        ],
    )
    def test_get_union_types(self, type_hint: Any, expected: set) -> None:
        """Test get_union_types function."""
        assert set(get_union_types(type_hint)) == expected

    # Test is_list_type
    @pytest.mark.parametrize(
        "type_hint,expected",
        [
            (List[int], True),
            (List[str], True),
            (List[Dict[str, int]], True),
            (int, False),
            (str, False),
            (Dict[str, int], False),
        ],
    )
    def test_is_list_type(self, type_hint: Any, expected: bool) -> None:
        """Test is_list_type function."""
        assert is_list_type(type_hint) == expected

    # Test get_list_item_type
    @pytest.mark.parametrize(
        "type_hint,expected",
        [
            (List[int], int),
            (List[str], str),
            (List[Dict[str, int]], Dict[str, int]),
        ],
    )
    def test_get_list_item_type(self, type_hint: Any, expected: type) -> None:
        """Test get_list_item_type function."""
        assert get_list_item_type(type_hint) == expected

    # Test is_dict_type
    @pytest.mark.parametrize(
        "type_hint,expected",
        [
            (Dict[str, int], True),
            (Dict[str, str], True),
            (Dict[str, List[int]], True),
            (int, False),
            (str, False),
            (List[int], False),
        ],
    )
    def test_is_dict_type(self, type_hint: Any, expected: bool) -> None:
        """Test is_dict_type function."""
        assert is_dict_type(type_hint) == expected

    # Test get_dict_types
    @pytest.mark.parametrize(
        "type_hint,expected_key,expected_value",
        [
            (Dict[str, int], str, int),
            (Dict[int, str], int, str),
            (Dict[str, List[int]], str, List[int]),
        ],
    )
    def test_get_dict_types(
        self, type_hint: Any, expected_key: type, expected_value: type
    ) -> None:
        """Test get_dict_types function."""
        key_type, value_type = get_dict_types(type_hint)
        assert key_type == expected_key
        assert value_type == expected_value

    # Test is_basic_type
    @pytest.mark.parametrize(
        "type_hint,expected",
        [
            (int, True),
            (float, True),
            (bool, True),
            (str, True),
            (List[int], False),
            (Dict[str, int], False),
            (Optional[int], False),
            (Union[int, str], False),
        ],
    )
    def test_is_basic_type(self, type_hint: Any, expected: bool) -> None:
        """Test is_basic_type function."""
        assert is_basic_type(type_hint) == expected

    # Test convert_string_to_type
    @pytest.mark.parametrize(
        "value,type_hint,expected",
        [
            ("123", int, 123),
            ("3.14", float, 3.14),
            ("true", bool, True),
            ("false", bool, False),
            ("hello", str, "hello"),
            ("[1, 2, 3]", List[int], [1, 2, 3]),
            ('{"a": 1, "b": 2}', Dict[str, int], {"a": 1, "b": 2}),
        ],
    )
    def test_convert_string_to_type(
        self, value: str, type_hint: type, expected: Any
    ) -> None:
        """Test convert_string_to_type function."""
        assert convert_string_to_type(value, type_hint) == expected

    # Test validate_type
    @pytest.mark.parametrize(
        "value,type_hint,expected",
        [
            (123, int, True),
            (3.14, float, True),
            (True, bool, True),
            ("hello", str, True),
            ([1, 2, 3], List[int], True),
            ({"a": 1, "b": 2}, Dict[str, int], True),
            ("not_an_int", int, False),
            ([1, 2, "3"], List[int], False),
            ({"a": 1, "b": "2"}, Dict[str, int], False),
        ],
    )
    def test_validate_type(self, value: Any, type_hint: type, expected: bool) -> None:
        """Test validate_type function."""
        assert validate_type(value, type_hint) == expected
