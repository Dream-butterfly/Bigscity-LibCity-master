import argparse
import importlib.util
import sys


DEPENDENCIES = {
    "core": [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("scipy", "scipy"),
    ],
    "optional": [
        ("optuna", "optuna"),
        ("gensim", "gensim"),
        ("torch_geometric", "torch-geometric"),
        ("tslearn", "tslearn"),
        ("dtaidistance", "dtaidistance"),
        ("infomap", "infomap"),
        ("geopy", "geopy"),
        ("aiohttp", "aiohttp"),
    ],
}


def _check_module(module_name):
    return importlib.util.find_spec(module_name) is not None


def main():
    parser = argparse.ArgumentParser(description="Check availability of project dependencies.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero if any optional dependency is missing",
    )
    args = parser.parse_args()

    missing_core = []
    missing_optional = []

    for group, dependencies in DEPENDENCIES.items():
        print(f"[{group}]")
        for module_name, display_name in dependencies:
            installed = _check_module(module_name)
            status = "ok" if installed else "missing"
            print(f"- {display_name}: {status}")
            if not installed:
                if group == "core":
                    missing_core.append(display_name)
                else:
                    missing_optional.append(display_name)

    if missing_core:
        print(f"missing core dependencies: {', '.join(missing_core)}")
        sys.exit(1)
    if args.strict and missing_optional:
        print(f"missing optional dependencies: {', '.join(missing_optional)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
