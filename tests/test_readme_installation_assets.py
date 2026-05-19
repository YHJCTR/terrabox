from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_perception_installation_assets_are_documented_and_present():
    readme = (ROOT / "README.md").read_text()

    required_paths = [
        "agent_config.example.yaml",
        "scripts/download_weights.sh",
        "scripts/download_instructsam_models.py",
        "docker/sam2/build.sh",
        "docker/remoteclip/Dockerfile",
        "docker/remotesam/build.sh",
        "docker/strip_rcnn/build.sh",
        "docker/instructsam/build.sh",
    ]

    for relative_path in required_paths:
        assert (ROOT / relative_path).exists(), relative_path
        assert relative_path in readme


def test_readme_mentions_toolkit_sources_without_enabling_removed_drone_tools():
    readme = (ROOT / "README.md").read_text()

    assert "EarthAgent" in readme
    assert "OpenEarthAgent" in readme
    assert "drone_video" not in readme
