"""Data contracts and schema validation models for Analytics FC lakehouse.

Defines Pydantic V2 contracts enforcing attribute bounds, non-null composite
identifiers, valid age ranges, and data integrity across historical editions.
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlayerSeasonRecord(BaseModel):
    """Pydantic V2 data contract for an individual player-season record."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    player_id: int = Field(gt=0, description="Unique player identifier")
    fifa_version: int = Field(ge=7, le=26, description="FIFA or FC release edition year")
    short_name: str = Field(min_length=1, description="Common player display name")
    age: int = Field(ge=15, le=55, description="Player age in years")
    overall: int = Field(ge=35, le=99, description="Overall skill rating bounded in [35, 99]")
    potential: int = Field(ge=35, le=99, description="Potential ceiling rating bounded in [35, 99]")

    pace: float | None = Field(default=None, ge=10.0, le=99.0)
    shooting: float | None = Field(default=None, ge=10.0, le=99.0)
    passing: float | None = Field(default=None, ge=10.0, le=99.0)
    dribbling: float | None = Field(default=None, ge=10.0, le=99.0)
    defending: float | None = Field(default=None, ge=10.0, le=99.0)
    physic: float | None = Field(default=None, ge=10.0, le=99.0)

    value_eur: float | None = Field(default=None, ge=0.0)
    wage_eur: float | None = Field(default=None, ge=0.0)

    @field_validator("short_name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Player short_name cannot be empty or whitespace only")
        return cleaned


class SchemaValidationSummary(BaseModel):
    """Execution summary of schema validation checks across dataset batches."""

    total_records_checked: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    error_samples: list[str] = Field(default_factory=list)
    status: str = "PENDING"


def validate_season_dataframe(
    df: pd.DataFrame, max_rows: int | None = None
) -> SchemaValidationSummary:
    """Validate a pandas DataFrame of player-season records against Pydantic contract."""
    summary = SchemaValidationSummary()

    records_to_check = df if max_rows is None else df.head(max_rows)
    summary.total_records_checked = len(records_to_check)

    errors: list[str] = []
    valid_count = 0

    # Ensure required columns exist
    required_cols = ["player_id", "fifa_version", "short_name", "age", "overall", "potential"]
    missing = [col for col in required_cols if col not in records_to_check.columns]
    if missing:
        summary.status = "FAILED"
        summary.error_samples = [f"Missing required columns: {missing}"]
        return summary

    for idx, row in records_to_check.iterrows():
        try:
            row_dict = row.dropna().to_dict()
            PlayerSeasonRecord.model_validate(row_dict)
            valid_count += 1
        except Exception as exc:
            if len(errors) < 5:
                errors.append(f"Row {idx} invalid: {exc}")

    summary.valid_records = valid_count
    summary.invalid_records = summary.total_records_checked - valid_count
    summary.error_samples = errors
    summary.status = "PASSED" if summary.invalid_records == 0 else "FAILED"

    return summary
