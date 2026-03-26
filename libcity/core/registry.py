class Registry:
    def __init__(self, name):
        self.name = name
        self._items = {}

    def register(self, obj=None, *, name=None):
        def decorator(target):
            key = name or target.__name__
            if key in self._items:
                raise KeyError(f"{key} is already registered in {self.name}")
            self._items[key] = target
            return target

        if obj is None:
            return decorator
        return decorator(obj)

    def get(self, name):
        try:
            return self._items[name]
        except KeyError as exc:
            raise AttributeError(f"{name} is not registered in {self.name}") from exc

    def items(self):
        return dict(self._items)
