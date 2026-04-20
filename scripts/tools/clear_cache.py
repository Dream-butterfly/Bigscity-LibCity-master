import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "cache"


def _iter_targets(scope):
    if not CACHE_ROOT.exists():
        return []
    if scope == "all":
        return [path for path in CACHE_ROOT.iterdir()]
    if scope == "dataset":
        target = CACHE_ROOT / "dataset_cache"
        return [target] if target.exists() else []
    if scope == "exp":
        return [path for path in CACHE_ROOT.iterdir() if path.is_dir() and path.name != "dataset_cache"]
    raise ValueError(f"Unsupported scope: {scope}")


def _delete_path(path, dry_run):
    if dry_run:
        print(f"[dry-run] remove {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"removed {path}")


def main():
    parser = argparse.ArgumentParser(description="Clear GNNTP cache files.")
    parser.add_argument(
        "--scope",
        choices=["all", "dataset", "exp"],
        default="exp",
        help="which cache scope to clear",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be removed without deleting files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip interactive confirmation",
    )
    args = parser.parse_args()

    targets = _iter_targets(args.scope)
    if not targets:
        print("no cache targets found")
        return

    print("targets:")
    for target in targets:
        print(f"- {target}")

    if not args.dry_run and not args.yes:
        answer = input("Delete these cache targets? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("aborted")
            return

    for target in targets:
        _delete_path(target, args.dry_run)


if __name__ == "__main__":
    main()
