"""
Exception hierarchy for LensingSSC.

This module defines all custom exceptions used throughout the package,
providing clear error messages and proper inheritance structure.
"""

from typing import Optional, Any, Dict


class LensingSSCError(Exception):
    """Base exception for all LensingSSC errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


class ValidationError(LensingSSCError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(LensingSSCError):
    """Raised when configuration is invalid or missing."""
    pass


class ProviderError(LensingSSCError):
    """Raised when a provider operation fails."""
    pass


class ProcessingError(LensingSSCError):
    """Raised when a processing operation fails."""
    pass


class DataError(LensingSSCError):
    """Raised when data operations fail."""
    pass


class GeometryError(LensingSSCError):
    """Raised when geometric operations fail."""
    pass


class StatisticsError(LensingSSCError):
    """Raised when statistical calculations fail."""
    pass


class IOError(LensingSSCError):
    """Raised when input/output operations fail."""
    pass


class VisualizationError(LensingSSCError):
    """Raised when visualization operations fail."""
    pass


# Compatibility with existing code
class PreprocessingError(ProcessingError):
    """Legacy alias for ProcessingError."""
    pass