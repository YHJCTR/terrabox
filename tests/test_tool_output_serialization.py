import json

import numpy as np

from terrabox.core.utils.tool_output_serialization import make_json_safe


def test_make_json_safe_converts_numpy_scalars_and_arrays():
    payload = {
        "count": np.int64(7),
        "score": np.float32(0.5),
        "items": [np.int32(2), np.array([1, 2, 3], dtype=np.int64)],
    }

    safe = make_json_safe(payload)

    assert safe == {"count": 7, "score": 0.5, "items": [2, [1, 2, 3]]}
    json.dumps(safe)
