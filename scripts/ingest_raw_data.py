import duckdb
import os

db_path = os.path.join('dbt_project', 'analytics_fc.duckdb')
con = duckdb.connect(db_path)

con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

print("Loading fifatable.csv...")
con.execute("CREATE OR REPLACE TABLE raw.fifatable AS SELECT * FROM read_csv_auto('data/raw/fifatable.csv', ignore_errors=true);")

print("Loading tm_stats.csv...")
con.execute("CREATE OR REPLACE TABLE raw.tm_stats AS SELECT * FROM read_csv_auto('data/raw/tm_stats.csv', ignore_errors=true);")

print("Loading tm_trophies.csv...")
con.execute("CREATE OR REPLACE TABLE raw.tm_trophies AS SELECT * FROM read_csv_auto('data/raw/tm_trophies.csv', ignore_errors=true);")

print("Success! Data loaded into DuckDB Bronze Layer.")
con.close()
