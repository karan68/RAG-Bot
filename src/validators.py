"""
Input Validation
Validates and sanitizes user input before processing.
"""

import re
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    cleaned_input: str
    error_message: Optional[str] = None


class InputValidator:
    """Validates user input for the chatbot."""
    
    def __init__(
        self,
        max_length: int = 500,
        min_length: int = 2,
        blocked_patterns: list = None
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.blocked_patterns = blocked_patterns or [
            r'ignore.*instruction',
            r'forget.*everything',
            r'you are now',
            r'pretend to be',
            r'act as if',
            r'disregard.*above',
            r'system prompt',
            r'<\s*script',  # HTML/JS injection
            r'<\s*img',
        ]
        
        # Compile patterns for efficiency
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.blocked_patterns
        ]
    
    def validate(self, user_input: str) -> ValidationResult:
        """
        Validate user input.
        
        Returns:
            ValidationResult with is_valid, cleaned_input, and optional error_message
        """
        # Check for None or non-string
        if user_input is None:
            return ValidationResult(
                is_valid=False,
                cleaned_input="",
                error_message="Please enter a question."
            )
        
        if not isinstance(user_input, str):
            return ValidationResult(
                is_valid=False,
                cleaned_input="",
                error_message="Invalid input type."
            )
        
        # Clean whitespace
        cleaned = user_input.strip()
        
        # Check empty
        if not cleaned:
            return ValidationResult(
                is_valid=False,
                cleaned_input="",
                error_message="Please enter a question."
            )
        
        # Check minimum length
        if len(cleaned) < self.min_length:
            return ValidationResult(
                is_valid=False,
                cleaned_input=cleaned,
                error_message="Question is too short. Please be more specific."
            )
        
        # Check maximum length
        if len(cleaned) > self.max_length:
            return ValidationResult(
                is_valid=False,
                cleaned_input=cleaned[:self.max_length],
                error_message=f"Question is too long (max {self.max_length} characters). Please shorten it."
            )
        
        # Check for blocked patterns (prompt injection)
        for pattern in self._compiled_patterns:
            if pattern.search(cleaned):
                return ValidationResult(
                    is_valid=False,
                    cleaned_input=cleaned,
                    error_message="I can only answer questions about device specifications. Please rephrase your question."
                )
        
        # Remove potentially harmful characters but keep normal punctuation
        # Only remove control characters and null bytes
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
        
        # All checks passed
        return ValidationResult(
            is_valid=True,
            cleaned_input=cleaned,
            error_message=None
        )
    
    def sanitize_for_display(self, text: str) -> str:
        """
        Sanitize text for safe display in UI.
        Escapes HTML-like content.
        """
        if not text:
            return ""
        
        # Escape HTML special characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        return text


# Global validator instance with default settings
_validator = None


def get_validator() -> InputValidator:
    """Get the global validator instance."""
    global _validator
    if _validator is None:
        # Try to load settings from config
        try:
            from config import get_config
            config = get_config()
            _validator = InputValidator(
                max_length=config.get('validation.max_query_length', 500),
                min_length=config.get('validation.min_query_length', 2),
                blocked_patterns=config.get('validation.blocked_patterns', [])
            )
        except ImportError:
            _validator = InputValidator()
    
    return _validator


def validate_input(user_input: str) -> ValidationResult:
    """Convenience function to validate input."""
    return get_validator().validate(user_input)
