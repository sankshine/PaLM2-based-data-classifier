"""
Unit tests for PaLM 2 Classifier

Run with: pytest tests/test_palm2_classifier.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.classification.palm2_classifier import PaLM2Classifier, ClassificationResult


class TestPaLM2Classifier:
    """Test cases for PaLM2Classifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance for testing"""
        with patch('google.cloud.aiplatform.init'):
            with patch('src.classification.palm2_classifier.TextGenerationModel'):
                return PaLM2Classifier(
                    project_id="test-project",
                    location="us-central1"
                )
    
    def test_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier.project_id == "test-project"
        assert classifier.location == "us-central1"
        assert classifier.temperature == 0.1
        assert classifier.max_tokens == 256
    
    def test_build_classification_prompt(self, classifier):
        """Test prompt building"""
        field_metadata = {
            'field_name': 'email_address',
            'field_type': 'STRING',
            'description': 'Customer email',
            'table': 'customers',
            'dataset': 'crm'
        }
        
        field_profile = {
            'sample_count': 1000,
            'unique_count': 998,
            'cardinality_ratio': 0.998,
            'null_count': 2,
            'max_length': 50,
            'avg_length': 25.5,
            'patterns_detected': [
                {'pattern': 'email', 'confidence': 0.99}
            ]
        }
        
        prompt = classifier._build_classification_prompt(field_metadata, field_profile)
        
        assert 'email_address' in prompt
        assert 'STRING' in prompt
        assert 'Customer email' in prompt
        assert 'email' in prompt
        assert 'HSPII' in prompt
        assert 'PII' in prompt
        assert 'PHI' in prompt
    
    def test_parse_classification_response_valid(self, classifier):
        """Test parsing valid response"""
        response_text = '''
        {
            "classification": "PII",
            "confidence": 0.95,
            "reasoning": "Field contains email addresses"
        }
        '''
        
        result = classifier._parse_classification_response(response_text, "email_address")
        
        assert isinstance(result, ClassificationResult)
        assert result.field_name == "email_address"
        assert result.classification == "PII"
        assert result.confidence == 0.95
        assert "email addresses" in result.reasoning
    
    def test_parse_classification_response_invalid(self, classifier):
        """Test parsing invalid response"""
        response_text = "Invalid JSON response"
        
        result = classifier._parse_classification_response(response_text, "test_field")
        
        assert result.classification == "NON_SENSITIVE"
        assert result.confidence == 0.0
        assert "Failed to parse" in result.reasoning
    
    def test_classify_field_pii(self, classifier):
        """Test classification of PII field"""
        # Mock the model prediction
        mock_response = Mock()
        mock_response.text = '{"classification": "PII", "confidence": 0.92, "reasoning": "Email pattern detected"}'
        classifier.model.predict = Mock(return_value=mock_response)
        
        field_metadata = {
            'field_name': 'customer_email',
            'field_type': 'STRING',
            'description': 'Email address'
        }
        
        field_profile = {
            'patterns_detected': [{'pattern': 'email', 'confidence': 0.95}],
            'cardinality_ratio': 0.99
        }
        
        result = classifier.classify_field(field_metadata, field_profile)
        
        assert result.classification == "PII"
        assert result.confidence == 0.92
    
    def test_classify_field_hspii(self, classifier):
        """Test classification of HSPII field"""
        mock_response = Mock()
        mock_response.text = '{"classification": "HSPII", "confidence": 0.98, "reasoning": "SSN pattern detected"}'
        classifier.model.predict = Mock(return_value=mock_response)
        
        field_metadata = {
            'field_name': 'ssn',
            'field_type': 'STRING',
            'description': 'Social security number'
        }
        
        field_profile = {
            'patterns_detected': [{'pattern': 'ssn', 'confidence': 0.96}]
        }
        
        result = classifier.classify_field(field_metadata, field_profile)
        
        assert result.classification == "HSPII"
        assert result.confidence == 0.98
    
    def test_batch_classify(self, classifier):
        """Test batch classification"""
        mock_response = Mock()
        mock_response.text = '{"classification": "PII", "confidence": 0.90, "reasoning": "Test"}'
        classifier.model.predict = Mock(return_value=mock_response)
        
        fields = [
            {'field_name': 'email', 'field_type': 'STRING'},
            {'field_name': 'name', 'field_type': 'STRING'}
        ]
        
        profiles = {
            'email': {'patterns_detected': []},
            'name': {'patterns_detected': []}
        }
        
        results = classifier.batch_classify(fields, profiles)
        
        assert len(results) == 2
        assert all(isinstance(r, ClassificationResult) for r in results)
    
    def test_low_confidence_threshold(self, classifier):
        """Test low confidence detection"""
        mock_response = Mock()
        mock_response.text = '{"classification": "NON_SENSITIVE", "confidence": 0.60, "reasoning": "Uncertain"}'
        classifier.model.predict = Mock(return_value=mock_response)
        
        fields = [{'field_name': 'unknown_field', 'field_type': 'STRING'}]
        profiles = {'unknown_field': {}}
        
        results = classifier.batch_classify(fields, profiles, confidence_threshold=0.7)
        
        assert results[0].confidence < 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
