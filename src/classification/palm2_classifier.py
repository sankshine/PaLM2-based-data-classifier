"""
PaLM 2 Classification Module

Uses Vertex AI and PaLM 2 to classify data fields as HSPII, PII, PHI, 
or NON_SENSITIVE based on field metadata and profiling results.
"""

from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of field classification"""
    field_name: str
    classification: str  # HSPII, PII, PHI, NON_SENSITIVE
    confidence: float
    reasoning: str


class PaLM2Classifier:
    """Classifies data fields using PaLM 2 LLM"""
    
    def __init__(self, 
                 project_id: str, 
                 location: str, 
                 model_name: str = "text-bison@002",
                 temperature: float = 0.1,
                 max_tokens: int = 256):
        """
        Initialize PaLM 2 classifier
        
        Args:
            project_id: GCP project ID
            location: GCP location (e.g., 'us-central1')
            model_name: Vertex AI model name
            temperature: Sampling temperature (lower = more consistent)
            max_tokens: Maximum response tokens
        """
        aiplatform.init(project=project_id, location=location)
        self.model = TextGenerationModel.from_pretrained(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.project_id = project_id
        self.location = location
        
        logger.info(f"Initialized PaLM2Classifier with model {model_name}")
        
    def classify_field(self, 
                      field_metadata: Dict, 
                      field_profile: Dict) -> ClassificationResult:
        """
        Classify a single field using PaLM 2
        
        Args:
            field_metadata: Field information (name, type, description, etc.)
            field_profile: Statistical profile (patterns, cardinality, etc.)
            
        Returns:
            ClassificationResult with category, confidence, and reasoning
        """
        
        # Build context-rich prompt
        prompt = self._build_classification_prompt(field_metadata, field_profile)
        
        # Generate classification
        try:
            response = self.model.predict(
                prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_k=40,
                top_p=0.95
            )
            
            # Parse response
            result = self._parse_classification_response(
                response.text, 
                field_metadata['field_name']
            )
            
            logger.info(f"Classified {field_metadata['field_name']}: "
                       f"{result.classification} (confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying {field_metadata['field_name']}: {str(e)}")
            return ClassificationResult(
                field_name=field_metadata['field_name'],
                classification='NON_SENSITIVE',
                confidence=0.0,
                reasoning=f"Classification error: {str(e)}"
            )
    
    def _build_classification_prompt(self, 
                                    metadata: Dict, 
                                    profile: Dict) -> str:
        """
        Construct detailed prompt for classification
        
        Args:
            metadata: Field metadata
            profile: Field profile
            
        Returns:
            Formatted prompt string
        """
        
        patterns_str = ", ".join([
            f"{p['pattern']} ({p['confidence']:.2f})" 
            for p in profile.get('patterns_detected', [])
        ])
        
        sample_count = profile.get('sample_count', 0)
        null_pct = (profile.get('null_count', 0) / max(sample_count, 1)) * 100
        
        prompt = f"""You are a data classification expert specializing in PII, PHI, and HSPII identification.

Classify the following data field into one of these categories:
- HSPII (Highly Sensitive PII): SSN, credit card, passport, driver's license, biometric data, account credentials
- PII (Personally Identifiable Information): name, email, phone, address, DOB, IP address, device ID
- PHI (Protected Health Information): medical record number, diagnosis, prescription, treatment, health insurance ID
- NON_SENSITIVE: Other business data not personally identifiable

Field Information:
- Field Name: {metadata.get('field_name', 'N/A')}
- Data Type: {metadata.get('field_type', 'N/A')}
- Description: {metadata.get('description', 'N/A')}
- Table: {metadata.get('table', 'N/A')}
- Dataset: {metadata.get('dataset', 'N/A')}

Field Profile:
- Sample Count: {sample_count}
- Unique Values: {profile.get('unique_count', 0)}
- Cardinality Ratio: {profile.get('cardinality_ratio', 0):.2f}
- Null Percentage: {null_pct:.1f}%
- Max Length: {profile.get('max_length', 0)}
- Avg Length: {profile.get('avg_length', 0):.1f}
- Patterns Detected: {patterns_str or 'None'}

Provide your response in this exact JSON format:
{{
  "classification": "<HSPII|PII|PHI|NON_SENSITIVE>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

Consider:
1. Field naming conventions (e.g., 'ssn', 'email', 'patient_id')
2. Detected patterns (email, phone, SSN formats)
3. Data characteristics (cardinality, length, uniqueness)
4. Context from table/dataset names

Your classification:"""
        
        return prompt
    
    def _parse_classification_response(self, 
                                      response_text: str, 
                                      field_name: str) -> ClassificationResult:
        """
        Parse LLM response into structured result
        
        Args:
            response_text: Raw LLM response
            field_name: Name of the field being classified
            
        Returns:
            ClassificationResult object
        """
        
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Validate classification
            valid_classifications = ['HSPII', 'PII', 'PHI', 'NON_SENSITIVE']
            classification = parsed['classification'].upper()
            
            if classification not in valid_classifications:
                logger.warning(f"Invalid classification {classification}, defaulting to NON_SENSITIVE")
                classification = 'NON_SENSITIVE'
            
            return ClassificationResult(
                field_name=field_name,
                classification=classification,
                confidence=float(parsed.get('confidence', 0.0)),
                reasoning=parsed.get('reasoning', 'No reasoning provided')
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback for parsing errors
            logger.error(f"Failed to parse response for {field_name}: {str(e)}")
            return ClassificationResult(
                field_name=field_name,
                classification='NON_SENSITIVE',
                confidence=0.0,
                reasoning=f"Failed to parse response: {str(e)}"
            )
    
    def batch_classify(self, 
                      fields: List[Dict], 
                      profiles: Dict[str, Dict],
                      confidence_threshold: float = 0.7) -> List[ClassificationResult]:
        """
        Classify multiple fields
        
        Args:
            fields: List of field metadata dictionaries
            profiles: Dictionary mapping field names to profiles
            confidence_threshold: Minimum confidence to accept classification
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        low_confidence_count = 0
        
        for field in fields:
            field_name = field['field_name']
            profile = profiles.get(field_name, {})
            
            result = self.classify_field(field, profile)
            results.append(result)
            
            # Track low confidence cases
            if result.confidence < confidence_threshold:
                low_confidence_count += 1
                logger.warning(f"Low confidence classification for {field_name}: "
                             f"{result.confidence:.2f}")
        
        logger.info(f"Classified {len(results)} fields. "
                   f"{low_confidence_count} below confidence threshold.")
        
        return results
    
    def export_results_to_bigquery(self, 
                                   results: List[ClassificationResult],
                                   output_table: str) -> None:
        """
        Export classification results to BigQuery
        
        Args:
            results: List of classification results
            output_table: BigQuery table (project.dataset.table)
        """
        from google.cloud import bigquery
        import pandas as pd
        
        client = bigquery.Client(project=self.project_id)
        
        # Convert to DataFrame
        data = [{
            'field_name': r.field_name,
            'classification': r.classification,
            'confidence': r.confidence,
            'reasoning': r.reasoning,
            'timestamp': pd.Timestamp.now()
        } for r in results]
        
        df = pd.DataFrame(data)
        
        # Define schema
        schema = [
            bigquery.SchemaField("field_name", "STRING"),
            bigquery.SchemaField("classification", "STRING"),
            bigquery.SchemaField("confidence", "FLOAT"),
            bigquery.SchemaField("reasoning", "STRING"),
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
        ]
        
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        
        job = client.load_table_from_dataframe(df, output_table, job_config=job_config)
        job.result()
        
        logger.info(f"Exported {len(results)} classifications to {output_table}")


def main():
    """Example usage"""
    import os
    from google.cloud import bigquery
    
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION", "us-central1")
    
    # Initialize classifier
    classifier = PaLM2Classifier(
        project_id=project_id,
        location=location
    )
    
    # Load field metadata and profiles from BigQuery
    client = bigquery.Client(project=project_id)
    
    fields_query = """
        SELECT field_name, field_type, description, table_name, dataset_name
        FROM `project.dataset.field_metadata`
        LIMIT 100
    """
    fields = list(client.query(fields_query).result())
    
    profiles_query = """
        SELECT field_name, patterns_detected, cardinality_ratio, 
               sample_count, null_count, max_length, avg_length
        FROM `project.dataset.field_profiles`
    """
    profiles_df = client.query(profiles_query).to_dataframe()
    profiles = profiles_df.set_index('field_name').to_dict('index')
    
    # Classify fields
    results = classifier.batch_classify(
        fields=[dict(f) for f in fields],
        profiles=profiles
    )
    
    # Export results
    classifier.export_results_to_bigquery(
        results=results,
        output_table=f"{project_id}.metadata.classifications"
    )


if __name__ == "__main__":
    main()
