"""
Tests for schema validation.

This module contains unit tests for the schema validation functionality,
including validation of configuration against Pydantic schemas.
"""

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, ValidationError as PydanticValidationError

# Import the actual implementation
from medvllm.medical.config.validation.schema import (
    validate_config_schema,
    get_required_fields
)
from medvllm.medical.config.validation.exceptions import (
    SchemaValidationError,
    FieldTypeError,
    FieldValueError,
    RequiredFieldError
)

# Test schemas
class NestedModel(BaseModel):
    """Nested model for testing."""
    name: str
    value: int
    optional: Optional[str] = None

class TestSchema(BaseModel):
    """Test schema for validation."""
    name: str
    age: int
    active: bool = True
    tags: List[str] = []
    nested: NestedModel
    optional_field: Optional[float] = None


class TestSchemaValidation:
    """Test cases for schema validation."""
    
    @pytest.fixture
    def valid_data(self) -> Dict[str, Any]:
        """Return valid test data."""
        return {
            "name": "test",
            "age": 30,
            "active": True,
            "tags": ["tag1", "tag2"],
            "nested": {
                "name": "nested",
                "value": 42
            },
            "optional_field": 3.14
        }
    
    def test_validate_config_schema_valid(self, valid_data: Dict[str, Any]) -> None:
        """Test validation of valid data against schema."""
        # Should not raise
        result = validate_config_schema(valid_data, TestSchema)
        assert isinstance(result, TestSchema)
        assert result.name == "test"
        assert result.nested.name == "nested"
    
    @pytest.mark.parametrize("missing_field,expected_error", [
        ("name", "name"),
        ("age", "age"),
        ("nested", "nested"),
    ])
    def test_validate_config_schema_missing_required_field(
        self, valid_data: Dict[str, Any], missing_field: str, expected_error: str
    ) -> None:
        """Test validation with missing required fields."""
        # Given
        data = valid_data.copy()
        data.pop(missing_field, None)
        
        # When/Then
        with pytest.raises(RequiredFieldError) as exc_info:
            validate_config_schema(data, TestSchema)
        
        assert expected_error in str(exc_info.value)
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type,expected_error_msg", [
        ("age", "not_an_int", FieldTypeError, "invalid type"),
        ("active", "not_a_bool", FieldTypeError, "invalid type"),
        ("tags", "not_a_list", FieldTypeError, "invalid type"),
        ("nested", {"name": 123, "value": "not_an_int"}, FieldTypeError, "invalid type"),
        ("age", -1, FieldValueError, "value is less than 0"),
    ])
    def test_validate_config_schema_invalid_values(
        self, valid_data: Dict[str, Any], field_name: str, invalid_value: Any,
        expected_error_type: type, expected_error_msg: str
    ) -> None:
        """Test validation with invalid field values."""
        # Given
        data = valid_data.copy()
        if "." in field_name:
            # Handle nested fields
            parts = field_name.split(".")
            current = data
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = invalid_value
        else:
            data[field_name] = invalid_value
        
        # When/Then
        with pytest.raises(expected_error_type) as exc_info:
            validate_config_schema(data, TestSchema)
        
        assert expected_error_msg in str(exc_info.value).lower()
    
    def test_validate_config_schema_optional_fields(self, valid_data: Dict[str, Any]) -> None:
        """Test that optional fields can be omitted."""
        # Given
        data = valid_data.copy()
        data.pop("optional_field")
        
        # When
        result = validate_config_schema(data, TestSchema)
        
        # Then
        assert result.optional_field is None
    
    def test_validate_config_schema_none_for_optional(self, valid_data: Dict[str, Any]) -> None:
        """Test that None is allowed for optional fields."""
        # Given
        data = valid_data.copy()
        data["optional_field"] = None
        
        # When
        result = validate_config_schema(data, TestSchema)
        
        # Then
        assert result.optional_field is None
    
    def test_validate_config_schema_already_validated(self) -> None:
        """Test that an already validated object is returned as-is."""
        # Given
        obj = TestSchema(
            name="test",
            age=30,
            nested=NestedModel(name="nested", value=42)
        )
        
        # When
        result = validate_config_schema(obj, TestSchema)
        
        # Then
        assert result is obj
    
    def test_get_required_fields(self) -> None:
        """Test getting required fields from a schema."""
        # When
        required = get_required_fields(TestSchema)
        
        # Then
        assert set(required.keys()) == {"name", "age", "nested"}
        assert required["name"] == str
        assert required["age"] == int
        assert required["nested"] == NestedModel
