import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib
import libcity.tasks.registry as reg

print('Before import, TASK_MODEL_REGISTRY keys:', list(reg.TASK_MODEL_REGISTRY.keys()))
try:
    importlib.import_module('libcity.tasks.traffic_speed_prediction.STGCN')
    print('Imported STGCN module')
except Exception as e:
    print('Import STGCN failed:', e)

print('After import, TASK_MODEL_REGISTRY keys:', list(reg.TASK_MODEL_REGISTRY.keys()))
for task, r in reg.TASK_MODEL_REGISTRY.items():
    try:
        items = r.items()
    except Exception as e:
        items = str(e)
    print('Task', task, '->', list(items.keys()) if isinstance(items, dict) else items)

print('\nRegistry module file:', getattr(reg, '__file__', None))
try:
    mod = importlib.import_module('libcity.tasks.registry')
    print('Imported registry module file:', getattr(mod, '__file__', None))
except Exception as e:
    print('Cannot import registry module:', e)

# Test decorator manually
from libcity.tasks.registry import register_model
@register_model('traffic_state_pred')
class _TmpModel:
    pass

print('\nAfter manual register, TASK_MODEL_REGISTRY keys:', list(reg.TASK_MODEL_REGISTRY.keys()))
for task, r in reg.TASK_MODEL_REGISTRY.items():
    print('Task', task, 'registered models:', list(r.items().keys()))

# Inspect register_model identity inside the STGCN module
try:
    m = importlib.import_module('libcity.tasks.traffic_speed_prediction.STGCN')
    print('\nSTGCN module file:', getattr(m, '__file__', None))
    print('register_model in STGCN module refers to:', m.register_model if hasattr(m, 'register_model') else 'no attribute')
    # If register_model was imported via `from ... import`, it's bound in module globals
    reg_func = m.__dict__.get('register_model')
    print('STGCN.register_model id:', id(reg_func), 'module:', getattr(reg_func, '__module__', None) if reg_func else None)
    print('reg.register_model id:', id(reg.register_model), 'module:', reg.register_model.__module__)
except Exception as e:
    print('Error inspecting STGCN module:', e)
# Now import the actual model submodule which defines the STGCN class
try:
    mm = importlib.import_module('libcity.tasks.traffic_speed_prediction.STGCN.model')
    print('\nImported submodule:', getattr(mm, '__file__', None))
except Exception as e:
    print('Failed to import STGCN.model:', e)

print('\nFinal TASK_MODEL_REGISTRY keys:', list(reg.TASK_MODEL_REGISTRY.keys()))
for task, r in reg.TASK_MODEL_REGISTRY.items():
    print('After submodule import - Task', task, 'registered models:', list(r.items().keys()))
