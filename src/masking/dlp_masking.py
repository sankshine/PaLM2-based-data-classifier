"""
DLP Masking Module

Generates DLP policies and applies data masking based on classification results.
"""

from google.cloud import dlp_v2
from google.cloud.dlp_v2 import types
from google.cloud import bigquery
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DLPPolicyGenerator:
    """Generate and apply DLP masking policies"""
    
    def __init__(self, project_id: str, kms_key_name: Optional[str] = None):
        """
        Initialize DLP policy generator
        
        Args:
            project_id: GCP project ID
            kms_key_name: KMS key for encryption (format: projects/.../cryptoKeys/...)
        """
        self.dlp_client = dlp_v2.DlpServiceClient()
        self.project_id = project_id
        self.parent = f"projects/{project_id}"
        self.kms_key_name = kms_key_name
        
    def create_deidentify_config(self, 
                                 classification: str,
                                 field_name: str,
                                 custom_key: Optional[bytes] = None) -> Optional[types.DeidentifyConfig]:
        """
        Create DLP deidentification config based on classification
        
        Args:
            classification: Field classification (HSPII, PII, PHI, NON_SENSITIVE)
            field_name: Name of the field
            custom_key: Optional custom encryption key
            
        Returns:
            DeidentifyConfig object or None for NON_SENSITIVE
        """
        
        if classification == "HSPII":
            # Crypto hash for highly sensitive data
            return types.DeidentifyConfig(
                record_transformations=types.RecordTransformations(
                    field_transformations=[
                        types.FieldTransformation(
                            fields=[types.FieldId(name=field_name)],
                            primitive_transformation=types.PrimitiveTransformation(
                                crypto_hash_config=types.CryptoHashConfig(
                                    crypto_key=types.CryptoKey(
                                        kms_wrapped=types.KmsWrappedCryptoKey(
                                            wrapped_key=custom_key or b"default_key",
                                            crypto_key_name=self.kms_key_name
                                        )
                                    ) if self.kms_key_name else types.CryptoKey(
                                        transient=types.TransientCryptoKey(
                                            name="hspii-temp-key"
                                        )
                                    )
                                )
                            )
                        )
                    ]
                )
            )
        
        elif classification == "PII":
            # Format-preserving encryption for PII
            return types.DeidentifyConfig(
                record_transformations=types.RecordTransformations(
                    field_transformations=[
                        types.FieldTransformation(
                            fields=[types.FieldId(name=field_name)],
                            primitive_transformation=types.PrimitiveTransformation(
                                crypto_replace_ffx_fpe_config=types.CryptoReplaceFfxFpeConfig(
                                    crypto_key=types.CryptoKey(
                                        kms_wrapped=types.KmsWrappedCryptoKey(
                                            wrapped_key=custom_key or b"default_key",
                                            crypto_key_name=self.kms_key_name
                                        )
                                    ) if self.kms_key_name else types.CryptoKey(
                                        transient=types.TransientCryptoKey(
                                            name="pii-temp-key"
                                        )
                                    ),
                                    common_alphabet=types.CharsToIgnore.NUMERIC
                                )
                            )
                        )
                    ]
                )
            )
        
        elif classification == "PHI":
            # Date shifting for PHI dates
            return types.DeidentifyConfig(
                record_transformations=types.RecordTransformations(
                    field_transformations=[
                        types.FieldTransformation(
                            fields=[types.FieldId(name=field_name)],
                            primitive_transformation=types.PrimitiveTransformation(
                                date_shift_config=types.DateShiftConfig(
                                    upper_bound_days=30,
                                    lower_bound_days=-30,
                                    context=types.FieldId(name=f"{field_name}_context")
                                )
                            )
                        )
                    ]
                )
            )
        
        else:  # NON_SENSITIVE
            return None
    
    def create_inspect_template(self, 
                               sensitive_fields: List[Dict],
                               template_id: str = "pii-phi-template") -> str:
        """
        Create DLP inspect template for classified fields
        
        Args:
            sensitive_fields: List of sensitive field metadata
            template_id: Unique template identifier
            
        Returns:
            Template resource name
        """
        
        info_types = []
        for field in sensitive_fields:
            # Map classification to DLP info types
            if field['classification'] == 'HSPII':
                info_types.extend([
                    {'name': 'US_SOCIAL_SECURITY_NUMBER'},
                    {'name': 'CREDIT_CARD_NUMBER'},
                    {'name': 'PASSPORT'},
                    {'name': 'US_BANK_ROUTING_MICR'},
                    {'name': 'US_DRIVERS_LICENSE_NUMBER'}
                ])
            elif field['classification'] == 'PII':
                info_types.extend([
                    {'name': 'EMAIL_ADDRESS'},
                    {'name': 'PHONE_NUMBER'},
                    {'name': 'PERSON_NAME'},
                    {'name': 'STREET_ADDRESS'},
                    {'name': 'DATE_OF_BIRTH'},
                    {'name': 'IP_ADDRESS'}
                ])
            elif field['classification'] == 'PHI':
                info_types.extend([
                    {'name': 'MEDICAL_RECORD_NUMBER'},
                    {'name': 'FDA_CODE'},
                    {'name': 'ICD9_CODE'},
                    {'name': 'ICD10_CODE'}
                ])
        
        # Remove duplicates
        unique_info_types = list({it['name']: it for it in info_types}.values())
        
        inspect_config = types.InspectConfig(
            info_types=unique_info_types,
            min_likelihood=types.Likelihood.POSSIBLE,
            include_quote=False,
            limits=types.InspectConfig.FindingLimits(
                max_findings_per_request=1000
            )
        )
        
        template = types.InspectTemplate(
            display_name=template_id,
            inspect_config=inspect_config
        )
        
        try:
            response = self.dlp_client.create_inspect_template(
                parent=self.parent,
                inspect_template=template,
                template_id=template_id
            )
            logger.info(f"Created inspect template: {response.name}")
            return response.name
        except Exception as e:
            logger.error(f"Error creating inspect template: {str(e)}")
            raise
    
    def deidentify_table(self, 
                        table: types.Table,
                        deidentify_configs: Dict[str, types.DeidentifyConfig]) -> types.Table:
        """
        Apply deidentification to table data
        
        Args:
            table: DLP Table object
            deidentify_configs: Map of field names to deidentify configs
            
        Returns:
            Deidentified table
        """
        
        # Combine all field transformations
        all_transformations = []
        
        for field_name, config in deidentify_configs.items():
            if config:
                all_transformations.extend(
                    config.record_transformations.field_transformations
                )
        
        if not all_transformations:
            logger.warning("No transformations to apply")
            return table
        
        combined_config = types.DeidentifyConfig(
            record_transformations=types.RecordTransformations(
                field_transformations=all_transformations
            )
        )
        
        request = types.DeidentifyContentRequest(
            parent=self.parent,
            deidentify_config=combined_config,
            item=types.ContentItem(table=table)
        )
        
        try:
            response = self.dlp_client.deidentify_content(request=request)
            logger.info(f"Deidentified table with {len(table.rows)} rows")
            return response.item.table
        except Exception as e:
            logger.error(f"Error deidentifying table: {str(e)}")
            raise
    
    def create_dlp_job(self,
                      input_table: str,
                      output_table: str,
                      classification_table: str,
                      job_id: str = "pii-masking-job") -> str:
        """
        Create DLP job for batch deidentification
        
        Args:
            input_table: Source BigQuery table
            output_table: Destination BigQuery table
            classification_table: Table with field classifications
            job_id: Unique job identifier
            
        Returns:
            Job resource name
        """
        
        # Load classifications
        bq_client = bigquery.Client(project=self.project_id)
        query = f"""
            SELECT field_name, classification
            FROM `{classification_table}`
            WHERE classification IN ('HSPII', 'PII', 'PHI')
        """
        classifications = bq_client.query(query).to_dataframe()
        
        # Build field transformations
        field_transformations = []
        
        for _, row in classifications.iterrows():
            config = self.create_deidentify_config(
                row['classification'],
                row['field_name']
            )
            if config:
                field_transformations.extend(
                    config.record_transformations.field_transformations
                )
        
        # Create DLP job config
        deidentify_config = types.DeidentifyConfig(
            record_transformations=types.RecordTransformations(
                field_transformations=field_transformations
            )
        )
        
        # BigQuery source
        storage_config = types.StorageConfig(
            big_query_options=types.BigQueryOptions(
                table_reference=types.BigQueryTable(
                    project_id=self.project_id,
                    dataset_id=input_table.split('.')[1],
                    table_id=input_table.split('.')[2]
                )
            )
        )
        
        # BigQuery destination
        output_config = types.OutputStorageConfig(
            table=types.BigQueryTable(
                project_id=self.project_id,
                dataset_id=output_table.split('.')[1],
                table_id=output_table.split('.')[2]
            ),
            output_schema=types.OutputStorageConfig.OutputSchema.ALL_COLUMNS
        )
        
        # Create job
        request = types.CreateDlpJobRequest(
            parent=self.parent,
            job_id=job_id,
            job=types.DlpJob(
                risk_details=None,
                deidentify_config=deidentify_config
            ),
            storage_config=storage_config,
            output_config=output_config
        )
        
        try:
            response = self.dlp_client.create_dlp_job(request=request)
            logger.info(f"Created DLP job: {response.name}")
            return response.name
        except Exception as e:
            logger.error(f"Error creating DLP job: {str(e)}")
            raise


def main():
    """Example usage"""
    import os
    
    project_id = os.getenv("GCP_PROJECT_ID")
    kms_key = os.getenv("KMS_KEY_NAME")
    
    policy_gen = DLPPolicyGenerator(
        project_id=project_id,
        kms_key_name=kms_key
    )
    
    # Load sensitive fields
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id)
    
    query = """
        SELECT field_name, classification
        FROM `project.metadata.classifications`
        WHERE classification IN ('HSPII', 'PII', 'PHI')
    """
    sensitive_fields = list(client.query(query).result())
    
    # Create inspect template
    template_name = policy_gen.create_inspect_template(
        sensitive_fields=[dict(f) for f in sensitive_fields]
    )
    
    # Create DLP job for batch masking
    job_name = policy_gen.create_dlp_job(
        input_table=f"{project_id}.raw_data.customer_table",
        output_table=f"{project_id}.masked_data.customer_table",
        classification_table=f"{project_id}.metadata.classifications"
    )
    
    print(f"DLP job created: {job_name}")


if __name__ == "__main__":
    main()
