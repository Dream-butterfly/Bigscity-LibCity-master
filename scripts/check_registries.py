import importlib
import traceback

mods = [
    'libcity.core.registry',
    'libcity.model.registry',
    'libcity.executor.registry',
    'libcity.evaluator.registry',
    'libcity.data.registry',
]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, 'OK')
    except Exception as e:
        print(m, 'ERR')
        traceback.print_exc()
