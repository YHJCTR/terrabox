import os

import numpy as np

from terrabox.toolkits.earth_sci import calculate_sea_ice_handler, microwave_prm_handler


def _write_tif(path, value):
    import rasterio
    from rasterio.transform import from_origin

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(np.full((4, 4), value, dtype=np.float32), 1)


def test_microwave_prm_accepts_frontend_ordered_path_list(tmp_path):
    h_path = tmp_path / "h.tif"
    v_path = tmp_path / "v.tif"
    out_path = tmp_path / "prm.tif"
    _write_tif(str(h_path), 240)
    _write_tif(str(v_path), 250)

    result = microwave_prm_handler(
        {
            "bt_paths": [str(h_path), str(v_path)],
            "parameter": "VWC",
            "output_path": str(out_path),
        },
        {"file_param_names": ["bt_paths.H", "bt_paths.V"]},
        None,
    )

    assert result["output_path"] == str(out_path)
    assert out_path.exists()


def test_calculate_sea_ice_accepts_frontend_ordered_path_list(tmp_path):
    paths = {
        "19H": tmp_path / "19h.tif",
        "19V": tmp_path / "19v.tif",
        "37H": tmp_path / "37h.tif",
        "37V": tmp_path / "37v.tif",
    }
    for key, path in paths.items():
        _write_tif(str(path), {"19H": 240, "19V": 250, "37H": 255, "37V": 260}[key])
    out_path = tmp_path / "sea_ice.tif"

    result = calculate_sea_ice_handler(
        {
            "bt_paths": [str(paths["19H"]), str(paths["19V"]), str(paths["37H"]), str(paths["37V"])],
            "output_path": str(out_path),
        },
        {"file_param_names": ["bt_paths.19H", "bt_paths.19V", "bt_paths.37H", "bt_paths.37V"]},
        None,
    )

    assert result["output_path"] == str(out_path)
    assert out_path.exists()
