"""
Training Data Generation Module

Prepares training data for fine-tuning PaLM 2 on organization-specific
classification patterns.
"""

import pandas as pd
from google.cloud import bigquery
from typing import List, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generate training data for PaLM 2 fine-tuning"""
    
    def __init__(self, project_id: str):
        """
        Initialize training data generator
        
        Args:
            project_id: GCP project ID
        """
        self.bq_client = bigquery.Client(project=project_id)
        self.project_id = project_id
        
    def generate_training_examples(self, 
                                   labeled_fields_table: str,
                                   profiles_table: str,
                                   output_jsonl: str,
                                   train_split: float = 0.8) -> Dict[str, int]:
        """
        Create JSONL training data from labeled examples
        
        Args:
            labeled_fields_table: Table with ground truth labels
            profiles_table: Table with field profiles
            output_jsonl: GCS path for output JSONL file
            train_split: Fraction of data for training (rest for validation)
            
        Returns:
            Dictionary with counts of train/validation examples
        """
        
        query = f"""
        SELECT 
            lf.field_name,
            lf.field_type,
            lf.description,
            lf.table_name,
            lf.dataset_name,
            lf.classification_label,  -- Ground truth
            p.sample_count,
            p.unique_count,
            p.cardinality_ratio,
            p.null_count,
            p.max_length,
            p.avg_length,
            p.patterns_detected
        FROM `{labeled_fields_table}` lf
        JOIN `{profiles_table}` p
        ON lf.field_name = p.field_name
        WHERE lf.classification_label IN ('HSPII', 'PII', 'PHI', 'NON_SENSITIVE')
          AND lf.verified = TRUE
        """
        
        logger.info("Loading labeled training data...")
        df = self.bq_client.query(query).to_dataframe()
        
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split train/validation
        split_idx = int(len(df) * train_split)
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        logger.info(f"Split data: {len(train_df)} train, {len(val_df)} validation")
        
        # Convert to training format
        train_examples = self._convert_to_jsonl(train_df)
        val_examples = self._convert_to_jsonl(val_df)
        
        # Save to GCS
        train_path = output_jsonl.replace('.jsonl', '_train.jsonl')
        val_path = output_jsonl.replace('.jsonl', '_val.jsonl')
        
        self._save_to_gcs(train_examples, train_path)
        self._save_to_gcs(val_examples, val_path)
        
        logger.info(f"Training data saved to {train_path}")
        logger.info(f"Validation data saved to {val_path}")
        
        return {
            'train_count': len(train_examples),
            'validation_count': len(val_examples)
        }
    
    def _convert_to_jsonl(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to JSONL training format
        
        Args:
            df: DataFrame with labeled examples
            
        Returns:
            List of training examples
        """
        training_examples = []
        
        for _, row in df.iterrows():
            # Build input prompt (same format as inference)
            input_text = self._build_prompt(row)
            
            # Build expected output
            output_text = json.dumps({
                "classification": row['classification_label'],
                "confidence": 1.0,  # Ground truth has full confidence
                "reasoning": f"Labeled as {row['classification_label']} in training data"
            })
            
            training_examples.append({
                "input_text": input_text,
                "output_text": output_text
            })
        
        return training_examples
    
    def _build_prompt(self, row: pd.Series) -> str:
        """
        Build classification prompt from row data
        
        Args:
            row: Row from DataFrame
            
        Returns:
            Formatted prompt string
        """
        patterns = row.get('patterns_detected', '')
        sample_count = row.get('sample_count', 0)
        null_pct = (row.get('null_count', 0) / max(sample_count, 1)) * 100
        
        return f"""You are a data classification expert specializing in PII, PHI, and HSPII identification.

Classify the following data field into one of these categories:
- HSPII (Highly Sensitive PII): SSN, credit card, passport, driver's license, biometric data
- PII (Personally Identifiable Information): name, email, phone, address, DOB, IP address
- PHI (Protected Health Information): medical record number, diagnosis, prescription, treatment, health insurance
- NON_SENSITIVE: Other business data not personally identifiable

Field Information:
- Field Name: {row.get('field_name', 'N/A')}
- Data Type: {row.get('field_type', 'N/A')}
- Description: {row.get('description', 'N/A')}
- Table: {row.get('table_name', 'N/A')}
- Dataset: {row.get('dataset_name', 'N/A')}

Field Profile:
- Sample Count: {sample_count}
- Unique Values: {row.get('unique_count', 0)}
- Cardinality Ratio: {row.get('cardinality_ratio', 0):.2f}
- Null Percentage: {null_pct:.1f}%
- Max Length: {row.get('max_length', 0)}
- Patterns Detected: {patterns or 'None'}

Provide your response in JSON format with classification, confidence, and reasoning.

Your classification:"""
    
    def _save_to_gcs(self, examples: List[Dict], gcs_path: str) -> None:
        """
        Save training examples to GCS as JSONL
        
        Args:
            examples: List of training examples
            gcs_path: GCS path (gs://bucket/path/file.jsonl)
        """
        from google.cloud import storage
        
        # Parse GCS path
        path_parts = gcs_path.replace('gs://', '').split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        # Convert to JSONL
        jsonl_content = '\n'.join([json.dumps(ex) for ex in examples])
        
        # Upload to GCS
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(jsonl_content, content_type='application/json')
        
        logger.info(f"Uploaded {len(examples)} examples to {gcs_path}")
    
    def validate_training_data(self, jsonl_path: str) -> Dict[str, any]:
        """
        Validate training data format and distribution
        
        Args:
            jsonl_path: Path to JSONL file
            
        Returns:
            Validation statistics
        """
        from google.cloud import storage
        from collections import Counter
        
        # Download from GCS
        path_parts = jsonl_path.replace('gs://', '').split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()
        
        # Parse examples
        examples = [json.loads(line) for line in content.strip().split('\n')]
        
        # Extract classifications
        classifications = []
        for ex in examples:
            try:
                output = json.loads(ex['output_text'])
                classifications.append(output['classification'])
            except:
                continue
        
        # Calculate statistics
        class_dist = Counter(classifications)
        
        stats = {
            'total_examples': len(examples),
            'class_distribution': dict(class_dist),
            'min_class_count': min(class_dist.values()) if class_dist else 0,
            'max_class_count': max(class_dist.values()) if class_dist else 0,
            'is_balanced': max(class_dist.values()) / min(class_dist.values()) < 2.0 if class_dist else False
        }
        
        logger.info(f"Training data validation:")
        logger.info(f"  Total examples: {stats['total_examples']}")
        logger.info(f"  Class distribution: {stats['class_distribution']}")
        logger.info(f"  Balanced: {stats['is_balanced']}")
        
        return stats


def main():
    """Example usage"""
    import os
    
    project_id = os.getenv("GCP_PROJECT_ID")
    
    generator = TrainingDataGenerator(project_id=project_id)
    
    # Generate training data
    stats = generator.generate_training_examples(
        labeled_fields_table=f"{project_id}.metadata.labeled_fields",
        profiles_table=f"{project_id}.metadata.field_profiles",
        output_jsonl=f"gs://{project_id}-ml/training/pii_classification.jsonl"
    )
    
    print(f"Generated {stats['train_count']} training examples")
    print(f"Generated {stats['validation_count']} validation examples")
    
    # Validate training data
    train_stats = generator.validate_training_data(
        f"gs://{project_id}-ml/training/pii_classification_train.jsonl"
    )


if __name__ == "__main__":
    main()
