"""
Exception hierarchy for LensingSSC.

This module defines all custom exceptions used throughout the package,
providing clear error messages and proper inheritance structure.
"""

from typing import Optional, Any, Dict, Union


class LensingSSCError(Exception):
    """Base exception for all LensingSSC errors.
    
    This is the root exception class that all other LensingSSC exceptions
    inherit from. It provides enhanced error reporting with optional
    context information.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 cause: Optional[Exception] = None):
        """Initialize LensingSSC error.
        
        Parameters
        ----------
        message : str
            Primary error message
        details : dict, optional
            Additional context information
        cause : Exception, optional
            Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = self.message
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg += f" (Details: {details_str})"
        
        if self.cause:
            base_msg += f" (Caused by: {self.cause})"
        
        return base_msg
    
    def add_detail(self, key: str, value: Any) -> "LensingSSCError":
        """Add detail information to the error.
        
        Parameters
        ----------
        key : str
            Detail key
        value : Any
            Detail value
            
        Returns
        -------
        LensingSSCError
            Self for method chaining
        """
        self.details[key] = value
        return self
    
    def get_detail(self, key: str, default: Any = None) -> Any:
        """Get detail information from the error.
        
        Parameters
        ----------
        key : str
            Detail key
        default : Any
            Default value if key not found
            
        Returns
        -------
        Any
            Detail value or default
        """
        return self.details.get(key, default)


class ValidationError(LensingSSCError):
    """Raised when data validation fails.
    
    This exception is used when input data doesn't meet the required
    criteria or format specifications.
    """
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, **kwargs):
        """Initialize validation error.
        
        Parameters
        ----------
        message : str
            Validation error message
        field : str, optional
            Name of the field that failed validation
        value : Any, optional
            Value that failed validation
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if field is not None:
            details['field'] = field
        if value is not None:
            details['value'] = value
        
        super().__init__(message, details=details, **kwargs)
        self.field = field
        self.value = value


class ConfigurationError(LensingSSCError):
    """Raised when configuration is invalid or missing.
    
    This exception is used for configuration-related errors such as
    missing required parameters, invalid values, or malformed config files.
    """
    
    def __init__(self, message: str, config_file: Optional[str] = None,
                 parameter: Optional[str] = None, **kwargs):
        """Initialize configuration error.
        
        Parameters
        ----------
        message : str
            Configuration error message
        config_file : str, optional
            Path to the configuration file with issues
        parameter : str, optional
            Name of the problematic parameter
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if config_file is not None:
            details['config_file'] = config_file
        if parameter is not None:
            details['parameter'] = parameter
        
        super().__init__(message, details=details, **kwargs)
        self.config_file = config_file
        self.parameter = parameter


class ProviderError(LensingSSCError):
    """Raised when a provider operation fails.
    
    This exception is used when external library providers (healpy, lenstools, etc.)
    encounter errors or are not available.
    """
    
    def __init__(self, message: str, provider: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize provider error.
        
        Parameters
        ----------
        message : str
            Provider error message
        provider : str, optional
            Name of the provider that failed
        operation : str, optional
            Operation that was being attempted
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if provider is not None:
            details['provider'] = provider
        if operation is not None:
            details['operation'] = operation
        
        super().__init__(message, details=details, **kwargs)
        self.provider = provider
        self.operation = operation


class ProcessingError(LensingSSCError):
    """Raised when a processing operation fails.
    
    This exception is used for errors during data processing pipelines,
    such as preprocessing, patch extraction, or statistical analysis.
    """
    
    def __init__(self, message: str, step: Optional[str] = None,
                 input_data: Optional[str] = None, **kwargs):
        """Initialize processing error.
        
        Parameters
        ----------
        message : str
            Processing error message
        step : str, optional
            Processing step that failed
        input_data : str, optional
            Description of input data
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if step is not None:
            details['step'] = step
        if input_data is not None:
            details['input_data'] = input_data
        
        super().__init__(message, details=details, **kwargs)
        self.step = step
        self.input_data = input_data


class DataError(LensingSSCError):
    """Raised when data operations fail.
    
    This exception is used for errors related to data handling,
    such as format issues, corruption, or incompatible data types.
    """
    
    def __init__(self, message: str, data_type: Optional[str] = None,
                 expected_format: Optional[str] = None, **kwargs):
        """Initialize data error.
        
        Parameters
        ----------
        message : str
            Data error message
        data_type : str, optional
            Type of data that caused the error
        expected_format : str, optional
            Expected data format
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if data_type is not None:
            details['data_type'] = data_type
        if expected_format is not None:
            details['expected_format'] = expected_format
        
        super().__init__(message, details=details, **kwargs)
        self.data_type = data_type
        self.expected_format = expected_format


class GeometryError(LensingSSCError):
    """Raised when geometric operations fail.
    
    This exception is used for errors in coordinate transformations,
    geometric calculations, or spatial operations.
    """
    
    def __init__(self, message: str, coordinate_system: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize geometry error.
        
        Parameters
        ----------
        message : str
            Geometry error message
        coordinate_system : str, optional
            Coordinate system involved in the error
        operation : str, optional
            Geometric operation that failed
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if coordinate_system is not None:
            details['coordinate_system'] = coordinate_system
        if operation is not None:
            details['operation'] = operation
        
        super().__init__(message, details=details, **kwargs)
        self.coordinate_system = coordinate_system
        self.operation = operation


class StatisticsError(LensingSSCError):
    """Raised when statistical calculations fail.
    
    This exception is used for errors in statistical analysis,
    such as insufficient data, numerical instabilities, or invalid parameters.
    """
    
    def __init__(self, message: str, statistic: Optional[str] = None,
                 sample_size: Optional[int] = None, **kwargs):
        """Initialize statistics error.
        
        Parameters
        ----------
        message : str
            Statistics error message
        statistic : str, optional
            Statistical method that failed
        sample_size : int, optional
            Size of the data sample
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if statistic is not None:
            details['statistic'] = statistic
        if sample_size is not None:
            details['sample_size'] = sample_size
        
        super().__init__(message, details=details, **kwargs)
        self.statistic = statistic
        self.sample_size = sample_size


class IOError(LensingSSCError):
    """Raised when input/output operations fail.
    
    This exception is used for file I/O errors, network issues,
    or problems with external data sources.
    """
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize I/O error.
        
        Parameters
        ----------
        message : str
            I/O error message
        file_path : str, optional
            Path to the file involved in the error
        operation : str, optional
            I/O operation that failed ('read', 'write', etc.)
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if file_path is not None:
            details['file_path'] = file_path
        if operation is not None:
            details['operation'] = operation
        
        super().__init__(message, details=details, **kwargs)
        self.file_path = file_path
        self.operation = operation


class VisualizationError(LensingSSCError):
    """Raised when visualization operations fail.
    
    This exception is used for errors in plotting, figure generation,
    or other visualization tasks.
    """
    
    def __init__(self, message: str, plot_type: Optional[str] = None,
                 backend: Optional[str] = None, **kwargs):
        """Initialize visualization error.
        
        Parameters
        ----------
        message : str
            Visualization error message
        plot_type : str, optional
            Type of plot being generated
        backend : str, optional
            Visualization backend being used
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if plot_type is not None:
            details['plot_type'] = plot_type
        if backend is not None:
            details['backend'] = backend
        
        super().__init__(message, details=details, **kwargs)
        self.plot_type = plot_type
        self.backend = backend


# Legacy compatibility aliases
class PreprocessingError(ProcessingError):
    """Legacy alias for ProcessingError.
    
    This exception is maintained for backward compatibility with
    existing code that uses the old name.
    """
    pass


# Utility functions for error handling
def reraise_with_context(exception: Exception, context: str, 
                        additional_details: Optional[Dict[str, Any]] = None) -> None:
    """Re-raise an exception with additional context.
    
    Parameters
    ----------
    exception : Exception
        Original exception
    context : str
        Additional context message
    additional_details : dict, optional
        Additional details to include
        
    Raises
    ------
    LensingSSCError
        Enhanced exception with context
    """
    if isinstance(exception, LensingSSCError):
        # Add context to existing LensingSSC error
        enhanced_message = f"{context}: {exception.message}"
        if additional_details:
            exception.details.update(additional_details)
        exception.message = enhanced_message
        raise exception
    else:
        # Wrap other exceptions in LensingSSCError
        message = f"{context}: {str(exception)}"
        details = additional_details or {}
        details['original_exception_type'] = type(exception).__name__
        raise LensingSSCError(message, details=details, cause=exception)


def validate_not_none(value: Any, name: str) -> Any:
    """Validate that a value is not None.
    
    Parameters
    ----------
    value : Any
        Value to check
    name : str
        Name of the parameter for error messages
        
    Returns
    -------
    Any
        The value if it's not None
        
    Raises
    ------
    ValidationError
        If value is None
    """
    if value is None:
        raise ValidationError(f"Parameter '{name}' cannot be None", field=name, value=value)
    return value


def validate_type(value: Any, expected_type: type, name: str) -> Any:
    """Validate that a value has the expected type.
    
    Parameters
    ----------
    value : Any
        Value to check
    expected_type : type
        Expected type
    name : str
        Name of the parameter for error messages
        
    Returns
    -------
    Any
        The value if it has the correct type
        
    Raises
    ------
    ValidationError
        If value doesn't have the expected type
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
            field=name, value=value
        )
    return value


def validate_positive(value: Union[int, float], name: str) -> Union[int, float]:
    """Validate that a numeric value is positive.
    
    Parameters
    ----------
    value : int or float
        Value to check
    name : str
        Name of the parameter for error messages
        
    Returns
    -------
    int or float
        The value if it's positive
        
    Raises
    ------
    ValidationError
        If value is not positive
    """
    if value <= 0:
        raise ValidationError(f"Parameter '{name}' must be positive, got {value}",
                            field=name, value=value)
    return value


def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]], 
                  max_val: Optional[Union[int, float]], name: str) -> Union[int, float]:
    """Validate that a value is within a specified range.
    
    Parameters
    ----------
    value : int or float
        Value to check
    min_val : int, float, or None
        Minimum allowed value
    max_val : int, float, or None
        Maximum allowed value
    name : str
        Name of the parameter for error messages
        
    Returns
    -------
    int or float
        The value if it's in range
        
    Raises
    ------
    ValidationError
        If value is out of range
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"Parameter '{name}' must be >= {min_val}, got {value}",
                            field=name, value=value)
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"Parameter '{name}' must be <= {max_val}, got {value}",
                            field=name, value=value)
    
    return value