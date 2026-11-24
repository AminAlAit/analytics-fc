"""Production monitoring and data drift detection module for Analytics FC."""

from __future__ import annotations

import pandas.core.dtypes.common as _pd_common


def ensure_pandas_evidently_compat() -> None:
    """Restore infer_dtype_from_object on pandas 3.x for evidently compatibility."""
    if not hasattr(_pd_common, "infer_dtype_from_object"):
        _pd_common.infer_dtype_from_object = (  # type: ignore[attr-defined]
            lambda dtype: _pd_common.pandas_dtype(dtype).type
        )


ensure_pandas_evidently_compat()
