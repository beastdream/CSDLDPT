"""Read RGB/global Color Moments descriptors from MySQL."""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.database import connection_scope


class FeatureRecord(TypedDict):
    """One image and its ordered 9D RGB/global descriptor."""

    image_id: int
    filename: str
    filepath: str
    category: str
    features: NDArray[np.float64]


RGB_GLOBAL_FEATURE_QUERY = """
SELECT
    i.image_id,
    i.filename,
    i.filepath,
    c.category_name,
    f.r_mean,
    f.r_std,
    f.r_skew,
    f.g_mean,
    f.g_std,
    f.g_skew,
    f.b_mean,
    f.b_std,
    f.b_skew
FROM images AS i
JOIN categories AS c
    ON i.category_id = c.category_id
JOIN color_moment_features AS f
    ON i.image_id = f.image_id
WHERE f.color_space = %s
  AND f.feature_type = %s
ORDER BY i.image_id
"""


def get_all_rgb_global_features() -> list[FeatureRecord]:
    """Return every RGB/global descriptor from MySQL ordered by image ID.

    The database is accessed read-only. Each feature vector is validated as a
    finite float64 NumPy array with shape ``(9,)``.
    """
    with connection_scope() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(RGB_GLOBAL_FEATURE_QUERY, ("RGB", "GLOBAL"))
            rows = cursor.fetchall()
        finally:
            cursor.close()

    if not rows:
        raise ValueError("MySQL returned no RGB/GLOBAL Color Moments descriptors.")

    records: list[FeatureRecord] = []
    for row in rows:
        features = np.asarray(row[4:13], dtype=np.float64)
        if features.shape != (9,):
            raise ValueError(
                f"Image ID {row[0]} has feature shape {features.shape}; expected (9,)."
            )
        if not np.all(np.isfinite(features)):
            raise ValueError(
                f"Image ID {row[0]} has RGB/GLOBAL features containing NaN or Inf."
            )

        records.append(
            {
                "image_id": int(row[0]),
                "filename": str(row[1]),
                "filepath": str(row[2]),
                "category": str(row[3]),
                "features": features,
            }
        )
    return records
