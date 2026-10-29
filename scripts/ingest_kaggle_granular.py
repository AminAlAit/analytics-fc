import os
import zipfile

import duckdb
import pandas as pd


def main():
    db_path = os.path.join("dbt_project", "analytics_fc.duckdb")
    con = duckdb.connect(db_path)

    kaggle_dir = os.path.join("data", "raw", "kaggle")
    target_files = {
        "players_15.csv": 2015,
        "players_16.csv": 2016,
        "players_17.csv": 2017,
        "players_18.csv": 2018,
        "players_19.csv": 2019,
        "players_20.csv": 2020,
        "players_21.csv": 2021,
        "players_22.csv": 2022,
    }

    target_zip = None
    for root, dirs, files in os.walk(kaggle_dir):
        for f in files:
            if f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                try:
                    with zipfile.ZipFile(zip_path, "r") as z:
                        if any(zinfo.endswith("players_22.csv") for zinfo in z.namelist()):
                            target_zip = zip_path
                            break
                except Exception:
                    pass
        if target_zip:
            break

    if not target_zip:
        print("Could not find the zip containing players_22.csv")
        return

    print(f"Found target dataset in: {target_zip}")

    dfs = []
    with zipfile.ZipFile(target_zip, "r") as z:
        for zinfo in z.namelist():
            basename = os.path.basename(zinfo)
            if basename in target_files:
                print(f"Extracting {basename} in memory...")
                with z.open(zinfo) as csv_file:
                    df = pd.read_csv(csv_file, low_memory=False)
                    df["fifa_year"] = target_files[basename]
                    dfs.append(df)

    if not dfs:
        print("No target CSVs were successfully extracted.")
        return

    print("Concatenating yearly data...")
    combined_df = pd.concat(dfs, ignore_index=True)

    # Standardize common ID column naming just in case
    if "sofifa_id" in combined_df.columns:
        combined_df.rename(columns={"sofifa_id": "player_id"}, inplace=True)

    print(f"Total granular records loaded: {len(combined_df):,}")

    print("Writing directly into DuckDB raw.fifa_granular...")
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    con.execute("CREATE OR REPLACE TABLE raw.fifa_granular AS SELECT * FROM combined_df")

    print("Success! Pipeline is ready for dbt.")
    con.close()


if __name__ == "__main__":
    main()
