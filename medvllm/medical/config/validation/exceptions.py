"""
Custom exceptions for configuration validation.
"""

class ValidationError(ValueError):
    """Raised when a configuration validation fails."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Name of the field that failed validation (optional)
            value: The value that failed validation (optional)
        """
        self.field = field
        self.value = value
        self.message = message
        super().__init__(self.message)


class SchemaValidationError(ValidationError):
    """Raised when a configuration schema validation fails."""
    pass


class VersionCompatibilityError(ValidationError):
    """Raised when there's a version compatibility issue."""
    pass


class RequiredFieldError(ValidationError):
    """Raised when a required field is missing."""
    pass


class FieldTypeError(ValidationError):
    """Raised when a field has an invalid type."""
    pass


class FieldValueError(ValidationError):
    """Raised when a field has an invalid value."""
    pass
