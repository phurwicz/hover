class TypedValueDict(dict):
    """
    A dict that only allows values of a certain type.
    """

    def __init__(self, type_, *args, **kwargs):
        self._type = type_
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        self.typecheck(value)
        super().__setitem__(key, value)

    def typecheck(self, value):
        if not isinstance(value, self._type):
            raise TypeError(f"Value must be of type {self._type}, got {type(value)}")

    def update(self, other):
        for _value in other.values():
            self.typecheck(_value)
        super().update(other)
