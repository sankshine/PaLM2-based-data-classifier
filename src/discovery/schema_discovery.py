"""
Schema Discovery Module

Discovers schemas from BigQuery, Cloud Storage, and other data sources,
then registers them in Dataplex Data Catalog.
"""

from google.cloud import bigquery, dataplex_v1
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaDiscovery:
    """Discovers and catalogs data schemas across GCP data sources"""
    
    def __init__(self, project_id: str, location: str):
        """
        Initialize schema discovery client
        
        Args:
            project_id: GCP project ID
            location: GCP location/region (e.g., 'us-central1')
        """
        self.bq_client = bigquery.Client(project=project_id)
        self.dataplex_client = dataplex_v1.DataplexServiceClient()
        self.project_id = project_id
        self.location = location
        logger.info(f"Initialized SchemaDiscovery for project {project_id}")
        
    def discover_bigquery_schemas(self, dataset_ids: List[str]) -> List[Dict]:
        """
        Extract schemas from BigQuery datasets
        
        Args:
            dataset_ids: List of BigQuery dataset IDs to scan
            
        Returns:
            List of field metadata dictionaries
        """
        schemas = []
        
        for dataset_id in dataset_ids:
            logger.info(f"Discovering schemas in dataset: {dataset_id}")
            dataset_ref = f"{self.project_id}.{dataset_id}"
            
            try:
                dataset = self.bq_client.get_dataset(dataset_ref)
                
                # Get all tables in dataset
                tables = self.bq_client.list_tables(dataset_ref)
                
                for table_ref in tables:
                    table = self.bq_client.get_table(table_ref)
                    logger.info(f"Processing table: {table.table_id}")
                    
                    # Extract field metadata
                    for field in table.schema:
                        field_metadata = {
                            'source': 'bigquery',
                            'dataset': dataset_id,
                            'table': table.table_id,
                            'field_name': field.name,
                            'field_type': field.field_type,
                            'mode': field.mode,
                            'description': field.description or '',
                            'nested_fields': len(field.fields) if field.fields else 0
                        }
                        schemas.append(field_metadata)
                        
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
                continue
                    
        logger.info(f"Discovered {len(schemas)} fields from BigQuery")
        return schemas
    
    def discover_gcs_schemas(self, bucket_names: List[str], 
                            file_patterns: Optional[List[str]] = None) -> List[Dict]:
        """
        Extract schemas from Cloud Storage Parquet/Avro files
        
        Args:
            bucket_names: List of GCS bucket names
            file_patterns: Optional file patterns to match (e.g., ['*.parquet'])
            
        Returns:
            List of field metadata dictionaries
        """
        from google.cloud import storage
        import pyarrow.parquet as pq
        
        storage_client = storage.Client(project=self.project_id)
        schemas = []
        
        for bucket_name in bucket_names:
            logger.info(f"Discovering schemas in bucket: {bucket_name}")
            bucket = storage_client.bucket(bucket_name)
            
            # List blobs matching patterns
            blobs = bucket.list_blobs()
            
            for blob in blobs:
                if blob.name.endswith('.parquet'):
                    try:
                        # Download and read schema
                        blob_data = blob.download_as_bytes()
                        table = pq.read_table(blob_data)
                        
                        for field in table.schema:
                            field_metadata = {
                                'source': 'gcs',
                                'bucket': bucket_name,
                                'file': blob.name,
                                'field_name': field.name,
                                'field_type': str(field.type),
                                'mode': 'NULLABLE',
                                'description': '',
                                'nested_fields': 0
                            }
                            schemas.append(field_metadata)
                            
                    except Exception as e:
                        logger.warning(f"Could not read schema from {blob.name}: {str(e)}")
                        continue
        
        logger.info(f"Discovered {len(schemas)} fields from GCS")
        return schemas
    
    def register_in_dataplex(self, schemas: List[Dict], 
                            lake_id: str, zone_id: str) -> int:
        """
        Register discovered schemas in Dataplex Data Catalog
        
        Args:
            schemas: List of field metadata dictionaries
            lake_id: Dataplex lake ID
            zone_id: Dataplex zone ID
            
        Returns:
            Number of entries successfully registered
        """
        parent = (f"projects/{self.project_id}/locations/{self.location}/"
                 f"lakes/{lake_id}/zones/{zone_id}")
        
        registered_count = 0
        
        for schema in schemas:
            try:
                # Create Dataplex Entry
                entry = dataplex_v1.Entry(
                    name=f"{schema['dataset']}.{schema['table']}.{schema['field_name']}",
                    entry_type="SCHEMA_FIELD",
                    aspects={
                        'data_type': schema['field_type'],
                        'source_system': schema['source']
                    }
                )
                
                # Register in catalog
                self.dataplex_client.create_entry(parent=parent, entry=entry)
                registered_count += 1
                
            except Exception as e:
                logger.warning(f"Could not register {schema['field_name']}: {str(e)}")
                continue
            
        logger.info(f"Registered {registered_count}/{len(schemas)} fields in Dataplex")
        return registered_count
    
    def export_schemas_to_bigquery(self, schemas: List[Dict], 
                                   output_table: str) -> None:
        """
        Export discovered schemas to BigQuery table
        
        Args:
            schemas: List of field metadata dictionaries
            output_table: Fully qualified table name (project.dataset.table)
        """
        import pandas as pd
        
        df = pd.DataFrame(schemas)
        
        # Define schema
        schema = [
            bigquery.SchemaField("source", "STRING"),
            bigquery.SchemaField("dataset", "STRING"),
            bigquery.SchemaField("table", "STRING"),
            bigquery.SchemaField("field_name", "STRING"),
            bigquery.SchemaField("field_type", "STRING"),
            bigquery.SchemaField("mode", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("nested_fields", "INTEGER"),
        ]
        
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        
        job = self.bq_client.load_table_from_dataframe(
            df, output_table, job_config=job_config
        )
        job.result()
        
        logger.info(f"Exported {len(schemas)} schemas to {output_table}")


def main():
    """Example usage"""
    import os
    
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION", "us-central1")
    
    discovery = SchemaDiscovery(project_id=project_id, location=location)
    
    # Discover BigQuery schemas
    schemas = discovery.discover_bigquery_schemas(
        dataset_ids=["dataset1", "dataset2"]
    )
    
    # Export to BigQuery
    discovery.export_schemas_to_bigquery(
        schemas=schemas,
        output_table=f"{project_id}.metadata.discovered_schemas"
    )
    
    # Register in Dataplex
    discovery.register_in_dataplex(
        schemas=schemas,
        lake_id="data-lake",
        zone_id="raw-zone"
    )


if __name__ == "__main__":
    main()
