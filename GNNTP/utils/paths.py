from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DATA_ROOT = PROJECT_ROOT / "resource_data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CACHE_ROOT = PROJECT_ROOT / "cache"


def resolve_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
