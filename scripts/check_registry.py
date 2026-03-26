import sys
sys.path.insert(0, '.')
from libcity_new.model import registry

def print_registered():
    try:
        items = registry.TASK_MODEL_REGISTRY['traffic_state_pred'].items()
        print('Pre-registered models:', list(items.keys()))
    except Exception as e:
        print('Error listing registry:', e)

    try:
        registry._bootstrap_traffic_speed_models()
        items = registry.TASK_MODEL_REGISTRY['traffic_state_pred'].items()
        print('After bootstrap models:', list(items.keys()))
    except Exception as e:
        print('Bootstrap error:', e)

if __name__ == '__main__':
    print_registered()

