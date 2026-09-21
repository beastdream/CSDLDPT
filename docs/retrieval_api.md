# Retrieval Engine API Documentation

The `src.retrieval` module provides a unified Content-Based Image Retrieval (CBIR) engine interface for downstream applications, Web UIs, and evaluation scripts.

---

## Quick Start Example

```python
from src.retrieval import retrieve

# Query using fused RGB+HSV+LAB Color Moments, Euclidean distance, and Z-score normalization
results = retrieve(
    query_image="data/processed/wang/horses/700.jpg",
    descriptor="rgb_hsv_lab",
    metric="euclidean",
    top_k=10,
    normalization="zscore"
)

# Print Top-K nearest neighbors
print(f"Query Image: {results['query']['filepath']}")
for item in results["results"]:
    print(f"Rank {item['rank']}: ID={item['image_id']} | Category={item['category']} | Distance={item['distance']:.4f}")
```

---

## Function Signature

```python
def retrieve(
    query_image: str | PathLike[str],
    descriptor: str = "rgb",
    metric: str = "euclidean",
    top_k: int = 10,
    normalization: str = "none"
) -> dict[str, Any]:
```

---

## Parameters Reference

| Parameter | Type | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `query_image` | `str \| PathLike` | Valid file path | *Required* | Absolute or project-relative path to the query image. Supports internal dataset images and external uploaded images. |
| `descriptor` | `str` | `"rgb"`, `"hsv"`, `"lab"`, `"rgb_hsv"`, `"rgb_lab"`, `"hsv_lab"`, `"rgb_hsv_lab"` | `"rgb"` | Feature descriptor or fused color spaces. Feature dimensions are 9D for single spaces, 18D for 2-space fusion, and 27D for 3-space fusion. |
| `metric` | `str` | `"euclidean"`, `"manhattan"`, `"cosine"` | `"euclidean"` | Vector distance metric used for ranking neighbors. |
| `top_k` | `int` | Integer `> 0` | `10` | Number of nearest neighbors to retrieve. |
| `normalization` | `str` | `"none"`, `"zscore"` | `"none"` | Corpus-wide Z-score normalization option (`z = (x - mean) / std`). |

---

## Return Value Schema

The function returns a Python dictionary with the following structure:

```json
{
  "query": {
    "filepath": "data/processed/wang/horses/700.jpg",
    "image_id": 700,
    "category": "horses",
    "features": "<NDArray shape=(27,) dtype=float64>"
  },
  "descriptor": "rgb_hsv_lab",
  "method": "RGB_HSV_LAB Color Moments",
  "metric": "euclidean",
  "normalization": "zscore",
  "top_k": 10,
  "results": [
    {
      "rank": 1,
      "image_id": 705,
      "filename": "705.jpg",
      "filepath": "data/processed/wang/horses/705.jpg",
      "category": "horses",
      "distance": 0.35412
    },
    {
      "rank": 2,
      "image_id": 712,
      "filename": "712.jpg",
      "filepath": "data/processed/wang/horses/712.jpg",
      "category": "horses",
      "distance": 0.48123
    }
  ]
}
```

---

## Query Modes Behavior

1. **Internal Dataset Query**:
   If the `query_image` path matches an existing dataset image, the retrieval engine automatically excludes the query image itself from the returned `results` list (self-query exclusion). `results["query"]["image_id"]` contains the integer image ID.

2. **External Query Image**:
   If `query_image` is an external uploaded file not present in the dataset, `results["query"]["image_id"]` and `results["query"]["category"]` are set to `None`. The engine extracts the query descriptor and computes distances against all dataset items without excluding any candidate.

---

## Exceptions & Error Handling

- **`FileNotFoundError`**: Raised if `query_image` file does not exist on disk.
- **`ValueError`**: Raised if:
  - `top_k <= 0` or is not an integer.
  - `descriptor` is unsupported.
  - `metric` is unsupported.
  - `normalization` is unsupported.
  - OpenCV cannot decode `query_image`.
