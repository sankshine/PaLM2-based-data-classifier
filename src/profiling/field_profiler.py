"""
Field Profiling Module

Analyzes data fields to extract patterns, statistics, and characteristics
for classification.
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import re
from typing import Dict, Any, List
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FieldProfiler(beam.DoFn):
    """Profile fields to extract patterns and statistics"""
    
    PATTERN_REGEX = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'phone': r'^\+?1?\d{10,14}$',
        'ssn': r'^\d{3}-\d{2}-\d{4}$',
        'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
        'date': r'^\d{4}-\d{2}-\d{2}',
        'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        'zip_code': r'^\d{5}(-\d{4})?$',
        'url': r'^https?://[^\s]+$',
        'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    }
    
    def process(self, element: Dict[str, Any]):
        """
        Process a batch of field samples
        
        Args:
            element: Dict with 'field_name', 'field_type', and 'samples' keys
            
        Yields:
            Field profile dictionary
        """
        field_name = element['field_name']
        field_type = element['field_type']
        samples = element.get('samples', [])
        
        # Initialize profile
        profile = {
            'field_name': field_name,
            'field_type': field_type,
            'sample_count': len(samples),
            'null_count': sum(1 for s in samples if s is None),
            'unique_count': len(set(str(s) for s in samples if s)),
            'patterns_detected': [],
            'cardinality_ratio': 0.0,
            'max_length': 0,
            'avg_length': 0.0,
            'min_length': 0,
            'common_prefixes': [],
            'common_suffixes': []
        }
        
        # Remove nulls for analysis
        non_null_samples = [s for s in samples if s is not None]
        
        if not non_null_samples:
            yield profile
            return
            
        # Calculate statistics
        str_samples = [str(s) for s in non_null_samples]
        lengths = [len(s) for s in str_samples]
        
        profile['max_length'] = max(lengths)
        profile['min_length'] = min(lengths)
        profile['avg_length'] = sum(lengths) / len(lengths)
        profile['cardinality_ratio'] = profile['unique_count'] / profile['sample_count']
        
        # Detect patterns
        for pattern_name, regex in self.PATTERN_REGEX.items():
            matches = sum(1 for s in str_samples if re.match(regex, s, re.IGNORECASE))
            match_rate = matches / len(str_samples)
            
            if match_rate > 0.8:  # 80% match threshold
                profile['patterns_detected'].append({
                    'pattern': pattern_name,
                    'confidence': match_rate
                })
        
        # Detect common prefixes/suffixes (for IDs)
        if profile['cardinality_ratio'] > 0.9:  # High cardinality
            # Check for ID patterns
            prefixes = Counter(s[:3] for s in str_samples if len(s) >= 3)
            most_common_prefix = prefixes.most_common(1)
            if most_common_prefix and most_common_prefix[0][1] / len(str_samples) > 0.5:
                profile['common_prefixes'].append({
                    'prefix': most_common_prefix[0][0],
                    'frequency': most_common_prefix[0][1] / len(str_samples)
                })
                profile['patterns_detected'].append({
                    'pattern': 'id_with_prefix',
                    'confidence': most_common_prefix[0][1] / len(str_samples)
                })
            
            # Check suffixes
            suffixes = Counter(s[-3:] for s in str_samples if len(s) >= 3)
            most_common_suffix = suffixes.most_common(1)
            if most_common_suffix and most_common_suffix[0][1] / len(str_samples) > 0.5:
                profile['common_suffixes'].append({
                    'suffix': most_common_suffix[0][0],
                    'frequency': most_common_suffix[0][1] / len(str_samples)
                })
        
        # Check for numeric patterns
        if field_type in ['INTEGER', 'INT64', 'FLOAT', 'NUMERIC']:
            try:
                numeric_samples = [float(s) for s in str_samples]
                profile['min_value'] = min(numeric_samples)
                profile['max_value'] = max(numeric_samples)
                profile['avg_value'] = sum(numeric_samples) / len(numeric_samples)
            except ValueError:
                pass
        
        yield profile


class FieldSampler(beam.DoFn):
    """Sample field values for profiling"""
    
    def __init__(self, sample_size: int = 10000):
        self.sample_size = sample_size
    
    def process(self, element: Dict[str, Any]):
        """
        Sample field values from dataset
        
        Args:
            element: Dict with field information
            
        Yields:
            Field with samples
        """
        # In practice, this would query the actual data source
        # This is a placeholder for the sampling logic
        yield element


def run_profiling_pipeline(input_table: str, 
                          output_table: str,
                          project: str,
                          region: str,
                          runner: str = 'DataflowRunner',
                          temp_location: str = None,
                          staging_location: str = None) -> None:
    """
    Execute field profiling pipeline
    
    Args:
        input_table: BigQuery table with field samples (project.dataset.table)
        output_table: BigQuery table for profile results
        project: GCP project ID
        region: GCP region
        runner: Beam runner (DataflowRunner or DirectRunner for local)
        temp_location: GCS path for temp files
        staging_location: GCS path for staging
    """
    
    pipeline_options = PipelineOptions(
        runner=runner,
        project=project,
        region=region,
        temp_location=temp_location or f'gs://{project}-temp/profiling/temp',
        staging_location=staging_location or f'gs://{project}-temp/profiling/staging',
        save_main_session=True
    )
    
    # BigQuery schema for output
    output_schema = {
        'fields': [
            {'name': 'field_name', 'type': 'STRING'},
            {'name': 'field_type', 'type': 'STRING'},
            {'name': 'sample_count', 'type': 'INTEGER'},
            {'name': 'null_count', 'type': 'INTEGER'},
            {'name': 'unique_count', 'type': 'INTEGER'},
            {'name': 'patterns_detected', 'type': 'STRING'},
            {'name': 'cardinality_ratio', 'type': 'FLOAT'},
            {'name': 'max_length', 'type': 'INTEGER'},
            {'name': 'avg_length', 'type': 'FLOAT'},
            {'name': 'min_length', 'type': 'INTEGER'}
        ]
    }
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        profiles = (
            pipeline
            | 'Read Field Samples' >> beam.io.ReadFromBigQuery(
                query=f"""
                    SELECT 
                        field_name,
                        field_type,
                        ARRAY_AGG(field_value LIMIT 10000) as samples
                    FROM `{input_table}`
                    GROUP BY field_name, field_type
                """,
                use_standard_sql=True)
            | 'Profile Fields' >> beam.ParDo(FieldProfiler())
            | 'Format for BQ' >> beam.Map(lambda x: {
                'field_name': x['field_name'],
                'field_type': x['field_type'],
                'sample_count': x['sample_count'],
                'null_count': x['null_count'],
                'unique_count': x['unique_count'],
                'patterns_detected': str(x['patterns_detected']),
                'cardinality_ratio': x['cardinality_ratio'],
                'max_length': x['max_length'],
                'avg_length': x['avg_length'],
                'min_length': x['min_length']
            })
            | 'Write Profiles' >> beam.io.WriteToBigQuery(
                output_table,
                schema=output_schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)
        )
    
    logger.info(f"Profiling pipeline completed. Results written to {output_table}")


def main():
    """Example usage"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Run field profiling pipeline')
    parser.add_argument('--input-table', required=True, help='Input BigQuery table')
    parser.add_argument('--output-table', required=True, help='Output BigQuery table')
    parser.add_argument('--project', default=os.getenv('GCP_PROJECT_ID'), help='GCP project')
    parser.add_argument('--region', default=os.getenv('GCP_REGION', 'us-central1'), help='GCP region')
    parser.add_argument('--runner', default='DataflowRunner', help='Beam runner')
    
    args = parser.parse_args()
    
    run_profiling_pipeline(
        input_table=args.input_table,
        output_table=args.output_table,
        project=args.project,
        region=args.region,
        runner=args.runner
    )


if __name__ == "__main__":
    main()
