"""
PaLM 2 Fine-Tuning Module

Fine-tunes PaLM 2 model on organization-specific classification patterns
and evaluates performance.
"""

from google.cloud import aiplatform
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import bigquery
from typing import Dict, List, Optional
import json
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaLM2FineTuner:
    """Fine-tune PaLM 2 on custom training data"""
    
    def __init__(self, project_id: str, location: str):
        """
        Initialize fine-tuner
        
        Args:
            project_id: GCP project ID
            location: GCP location (e.g., 'us-central1')
        """
        aiplatform.init(project=project_id, location=location)
        self.project_id = project_id
        self.location = location
        self.bq_client = bigquery.Client(project=project_id)
        
    def fine_tune_model(self, 
                       training_data_uri: str,
                       validation_data_uri: str,
                       model_display_name: str,
                       train_steps: int = 100,
                       learning_rate: float = 0.001,
                       base_model: str = "text-bison@002") -> str:
        """
        Execute fine-tuning job on Vertex AI
        
        Args:
            training_data_uri: GCS path to training JSONL
            validation_data_uri: GCS path to validation JSONL
            model_display_name: Display name for tuned model
            train_steps: Number of training steps
            learning_rate: Learning rate
            base_model: Base model to fine-tune
            
        Returns:
            Tuned model endpoint name
        """
        
        logger.info(f"Starting fine-tuning job for {model_display_name}")
        
        base_model_obj = TextGenerationModel.from_pretrained(base_model)
        
        # Fine-tuning configuration
        try:
            tuning_job = base_model_obj.tune_model(
                training_data=training_data_uri,
                validation_data=validation_data_uri,
                train_steps=train_steps,
                tuning_job_location=self.location,
                tuned_model_location=self.location,
                model_display_name=model_display_name,
                # Hyperparameters
                learning_rate=learning_rate,
                learning_rate_multiplier=1.0
            )
            
            logger.info(f"Fine-tuning job started: {tuning_job.resource_name}")
            logger.info("Waiting for completion (this may take 1-3 hours)...")
            
            # Wait for completion
            tuning_job.wait()
            
            logger.info(f"Fine-tuning completed!")
            logger.info(f"Tuned model endpoint: {tuning_job.tuned_model_endpoint_name}")
            
            return tuning_job.tuned_model_endpoint_name
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            raise
    
    def evaluate_model(self, 
                      model_endpoint: str,
                      test_data_table: str,
                      output_table: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate fine-tuned model performance
        
        Args:
            model_endpoint: Endpoint name of tuned model
            test_data_table: BigQuery table with test data
            output_table: Optional table to save predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        logger.info(f"Evaluating model: {model_endpoint}")
        
        # Load test data
        query = f"""
            SELECT 
                field_name, field_type, description, table_name, dataset_name,
                classification_label, sample_count, unique_count, cardinality_ratio,
                null_count, max_length, avg_length, patterns_detected
            FROM `{test_data_table}`
            WHERE classification_label IN ('HSPII', 'PII', 'PHI', 'NON_SENSITIVE')
        """
        test_df = self.bq_client.query(query).to_dataframe()
        
        logger.info(f"Loaded {len(test_df)} test examples")
        
        # Load fine-tuned model
        model = TextGenerationModel.get_tuned_model(model_endpoint)
        
        # Classify test examples
        predictions = []
        actuals = []
        confidences = []
        
        for idx, row in test_df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Evaluated {idx}/{len(test_df)} examples")
            
            prompt = self._build_prompt(row)
            
            try:
                response = model.predict(prompt, temperature=0.1, max_output_tokens=256)
                
                # Parse response
                json_start = response.text.find('{')
                json_end = response.text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    pred_data = json.loads(response.text[json_start:json_end])
                    pred = pred_data.get('classification', 'NON_SENSITIVE')
                    conf = pred_data.get('confidence', 0.0)
                else:
                    pred = 'NON_SENSITIVE'
                    conf = 0.0
                
                predictions.append(pred)
                confidences.append(conf)
                
            except Exception as e:
                logger.warning(f"Prediction failed for row {idx}: {str(e)}")
                predictions.append('NON_SENSITIVE')
                confidences.append(0.0)
            
            actuals.append(row['classification_label'])
        
        # Calculate metrics
        accuracy = accuracy_score(actuals, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actuals, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = \
            precision_recall_fscore_support(
                actuals, predictions, 
                labels=['HSPII', 'PII', 'PHI', 'NON_SENSITIVE'],
                zero_division=0
            )
        
        # Confusion matrix
        cm = confusion_matrix(
            actuals, predictions,
            labels=['HSPII', 'PII', 'PHI', 'NON_SENSITIVE']
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': sum(confidences) / len(confidences),
            'per_class_metrics': {
                'HSPII': {'precision': per_class_precision[0], 'recall': per_class_recall[0], 'f1': per_class_f1[0], 'support': support[0]},
                'PII': {'precision': per_class_precision[1], 'recall': per_class_recall[1], 'f1': per_class_f1[1], 'support': support[1]},
                'PHI': {'precision': per_class_precision[2], 'recall': per_class_recall[2], 'f1': per_class_f1[2], 'support': support[2]},
                'NON_SENSITIVE': {'precision': per_class_precision[3], 'recall': per_class_recall[3], 'f1': per_class_f1[3], 'support': support[3]}
            },
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"Model Evaluation Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Avg Confidence: {metrics['avg_confidence']:.4f}")
        
        # Save predictions if output table specified
        if output_table:
            self._save_predictions(
                test_df, predictions, confidences, output_table
            )
        
        return metrics
    
    def _build_prompt(self, row: pd.Series) -> str:
        """Build classification prompt from test row"""
        patterns = row.get('patterns_detected', '')
        sample_count = row.get('sample_count', 0)
        null_pct = (row.get('null_count', 0) / max(sample_count, 1)) * 100
        
        return f"""You are a data classification expert specializing in PII, PHI, and HSPII identification.

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

Classify this field as HSPII, PII, PHI, or NON_SENSITIVE in JSON format:"""
    
    def _save_predictions(self, 
                         test_df: pd.DataFrame,
                         predictions: List[str],
                         confidences: List[float],
                         output_table: str) -> None:
        """Save predictions to BigQuery"""
        
        results_df = test_df.copy()
        results_df['predicted_classification'] = predictions
        results_df['prediction_confidence'] = confidences
        results_df['correct'] = results_df['classification_label'] == results_df['predicted_classification']
        results_df['evaluation_timestamp'] = pd.Timestamp.now()
        
        # Save to BigQuery
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        
        job = self.bq_client.load_table_from_dataframe(
            results_df, output_table, job_config=job_config
        )
        job.result()
        
        logger.info(f"Saved {len(results_df)} predictions to {output_table}")
    
    def compare_models(self,
                      base_model_name: str,
                      tuned_model_endpoint: str,
                      test_data_table: str) -> Dict[str, Dict]:
        """
        Compare base model vs fine-tuned model
        
        Args:
            base_model_name: Base model identifier
            tuned_model_endpoint: Tuned model endpoint
            test_data_table: Test data table
            
        Returns:
            Comparison metrics
        """
        
        logger.info("Comparing base model vs fine-tuned model...")
        
        # Evaluate base model
        logger.info("Evaluating base model...")
        base_metrics = self.evaluate_model(
            base_model_name, 
            test_data_table
        )
        
        # Evaluate tuned model
        logger.info("Evaluating fine-tuned model...")
        tuned_metrics = self.evaluate_model(
            tuned_model_endpoint,
            test_data_table
        )
        
        # Calculate improvements
        improvements = {
            'accuracy_improvement': tuned_metrics['accuracy'] - base_metrics['accuracy'],
            'f1_improvement': tuned_metrics['f1_score'] - base_metrics['f1_score'],
            'precision_improvement': tuned_metrics['precision'] - base_metrics['precision'],
            'recall_improvement': tuned_metrics['recall'] - base_metrics['recall']
        }
        
        logger.info("Model Comparison:")
        logger.info(f"  Accuracy improvement: {improvements['accuracy_improvement']:+.4f}")
        logger.info(f"  F1 improvement: {improvements['f1_improvement']:+.4f}")
        
        return {
            'base_model': base_metrics,
            'tuned_model': tuned_metrics,
            'improvements': improvements
        }


def main():
    """Example usage"""
    import os
    
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION", "us-central1")
    
    tuner = PaLM2FineTuner(project_id=project_id, location=location)
    
    # Fine-tune model
    model_endpoint = tuner.fine_tune_model(
        training_data_uri=f"gs://{project_id}-ml/training/pii_classification_train.jsonl",
        validation_data_uri=f"gs://{project_id}-ml/training/pii_classification_val.jsonl",
        model_display_name="pii-classifier-v1",
        train_steps=100,
        learning_rate=0.001
    )
    
    # Evaluate model
    metrics = tuner.evaluate_model(
        model_endpoint=model_endpoint,
        test_data_table=f"{project_id}.metadata.test_fields",
        output_table=f"{project_id}.metadata.model_predictions"
    )
    
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
