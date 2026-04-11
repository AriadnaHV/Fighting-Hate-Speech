# ======================================= #
# Upload XLM-RoBERTa results to BigQuery  #
# Run from laptop                         #
# ======================================= #

import json
from google.cloud import bigquery
import datetime

# Load results from JSON file
with open('notebooks/phase1/xlm_roberta_results.json', 'r') as f:
    results = json.load(f)

bq_client = bigquery.Client()
table_id  = 'project-5c89dcac-34cb-453d-bd7.sinodio_results.xlm_roberta_results'

schema = [
    bigquery.SchemaField('model',      'STRING'),
    bigquery.SchemaField('split',      'STRING'),
    bigquery.SchemaField('accuracy',   'FLOAT'),
    bigquery.SchemaField('precision',  'FLOAT'),
    bigquery.SchemaField('recall',     'FLOAT'),
    bigquery.SchemaField('f1_macro',   'FLOAT'),
    bigquery.SchemaField('created_at', 'STRING'),
]

table = bigquery.Table(table_id, schema=schema)
table = bq_client.create_table(table, exists_ok=True)

errors = bq_client.insert_rows_json(table_id, results)

if not errors:
    print(f"Results written to BigQuery: {table_id}")
    for row in results:
        print(f"  {row['split']:<6} F1: {row['f1_macro']}")
else:
    print(f"BigQuery errors: {errors}")
