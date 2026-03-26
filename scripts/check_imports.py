#!/usr/bin/env python3
"""Lightweight import checker for the repo's `libcity` package.

Walks through submodules under `libcity` and attempts to import each one,
printing any traceback for failures. Intended for quick local validation.
"""
import pkgutil
import importlib
import traceback
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

package = 'libcity'

failures = []

try:
    pkg = importlib.import_module(package)
except Exception as e:
    print(f"Failed to import package {package}: {e}")
    traceback.print_exc()
    sys.exit(2)

for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=package + '.'):
    try:
        importlib.import_module(name)
    except Exception:
        print('\n' + '=' * 80)
        print('FAILED:', name)
        traceback.print_exc()
        failures.append(name)

print('\nScan complete. Modules scanned with import failures:')
for f in failures:
    print('-', f)

if failures:
    sys.exit(1)
else:
    print('No import failures detected.')
    sys.exit(0)
