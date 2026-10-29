import duckdb

con = duckdb.connect("dbt_project/analytics_fc.duckdb")
res = con.execute(
    "SELECT COUNT(*) FROM raw.fifatable WHERE minutes IS NOT NULL AND minutes != 'NA'"
).fetchall()
print(f"Valid minutes in fifatable: {res[0][0]}")
