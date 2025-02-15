from typing import Any


class RegistrantFactory:
    """
    To register the subclasses based on the name
    To be used as
    @RegistrantFactory.register("some_name")
    class SomeClass(RegistrantFactory)

    To build that class, one can use
    RegistrantFactory.create("some_name")
    """

    @classmethod
    def register(cls, subclass_name: str):
        def decorator(subclass: Any):
            subclass.subclasses[subclass_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, subclass_name: str, **params):
        if subclass_name not in cls.subclasses:
            raise ValueError("Unknown subclass name {}".format(subclass_name))
        print("-"*50)
        print(f"For class: {cls.__name__}, Selected subclass: ({subclass_name}):{cls.subclasses[subclass_name]}")
        print("-"*50)

        return cls.subclasses[subclass_name](**params)
