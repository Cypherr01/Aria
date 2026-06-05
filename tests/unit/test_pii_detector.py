import pytest
from responsible_ai.pii_detector import PIIDetector

def test_pii_detector_long_string_redaction():
    detector = PIIDetector()
    
    # Create a string longer than 1000 characters using normal sentences 
    # so spaCy's NER has good context.
    prefix = "This is a normal sentence to pad the string length. " * 22
    # 22 * 52 = 1144 characters
    
    test_str = prefix + "My name is John Doe and I work at Microsoft."
    
    # Run redaction
    redacted = detector.redact_for_logging(test_str)
    
    # Check that NER redaction worked past the 1000 character limit
    if detector.nlp is not None:
        assert "[REDACTED:PERSON]" in redacted
        assert "John Doe" not in redacted
        assert "[REDACTED:ORG]" in redacted
        assert "Microsoft" not in redacted
    else:
        pytest.skip("spaCy or en_core_web_sm not available, skipping NER test")

def test_pii_detector_regex_redaction():
    detector = PIIDetector()
    
    # Regex shouldn't be affected by chunking (it runs before chunking, on the whole string)
    test_str = ("A" * 1200) + " My email is test@example.com."
    redacted = detector.redact_for_logging(test_str)
    
    assert "[REDACTED:EMAIL]" in redacted
    assert "test@example.com" not in redacted


def test_pii_redacted_after_1000_characters():
    detector = PIIDetector()
    
    # Create a string of 1500 characters where email appears at position 1200
    padding = "A" * 1200
    text = padding + "My email address is test@example.com. Extra: " + ("B" * 200)
    
    redacted = detector.redact_for_logging(text)
    
    assert "test@example.com" not in redacted
    assert "[REDACTED:EMAIL]" in redacted


def test_pii_redaction_preserves_non_pii_content():
    detector = PIIDetector()
    
    # Create a string with PII at position 1200 and safe content at position 50
    padding1 = "A" * 50
    safe_content = "This is a safe sentence."
    padding2 = "A" * (1200 - len(padding1) - len(safe_content))
    text = padding1 + safe_content + padding2 + "My email is test@example.com. Extra: " + ("B" * 200)
    
    redacted = detector.redact_for_logging(text)
    
    assert safe_content in redacted
    assert "test@example.com" not in redacted
    assert "[REDACTED:EMAIL]" in redacted

