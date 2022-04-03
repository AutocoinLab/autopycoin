"""
Defines the base class for every classes in this library.
Most of the functions have been taken from tensorflow nightly.
"""

import abc
import typing
from inspect import signature, _empty

from tensorflow.python.util import tf_decorator

from .baseclass_field import AutopycoinField, _convert_value


NOT_SUPPORTED_FIELDS = ["return", "kwargs", "args"]


def dummy_callable(*args, **kwargs):
    """dummy callable"""
    pass


def _pop_field(type_hints):
    """Field not currently supported."""
    for hint in NOT_SUPPORTED_FIELDS:
        type_hints.pop(hint, None)


def _wrap_user_constructor(cls, attribute_name, attribute_value):
    """Wraps a user-defined constructor for autopycoin subclass `cls`."""

    def wrapped_attribute(self, *args, **kwargs):
        list_fields = cls._type_fields(  # pylint: disable=protected-access
            attribute_name, attribute_value
        )
        args, kwargs = convert_value(
            list_fields[f"{cls}_{attribute_name}"], attribute_name, *args, **kwargs
        )
        output = attribute_value(self, *args, **kwargs)
        self.__validate__(cls, output, attribute_name, *args, **kwargs)

        return output

    setattr(
        cls,
        attribute_name,
        tf_decorator.make_decorator(attribute_value, wrapped_attribute),
    )


def convert_value(list_fields, attribute_name, *args, **kwargs):
    """check type parameters to the expected types for no default parameters."""
    args = list(args)
    for idx, field in enumerate(list_fields):
        if not field.name in kwargs and idx <= len(args) - 1:
            args[idx] = _convert_value(
                args[idx],
                field.value_type,
                (f"value for {field.name} in {attribute_name}",),
            )
        elif field.name in kwargs:
            kwargs[field.name] = _convert_value(
                kwargs[field.name],
                field.value_type,
                (f"value for {field.name} in {attribute_name}",),
            )

    return args, kwargs


def _methods_to_inspect(cls):
    """Filter each method to inspect of autopycoin subclass `cls`."""
    for attribute_name in dir(cls):
        attribute_value = getattr(cls, attribute_name)
        # Check that it is callable
        if (
            callable(attribute_value)
            and not attribute_name in cls.NOT_INSPECT
            and not attribute_name.startswith("__")
            and not attribute_name.endswith("__")
            and not attribute_name.startswith("_abc_")
            and not attribute_name.startswith("_val")
            and attribute_name in cls.__dict__
        ) or attribute_name == "__init__":
            yield attribute_name, attribute_value


def _check_field_annotations(attribute_name, attribute_value):
    """Validates the field annotations for autopycoin subclass `cls`."""
    actual = set(signature(attribute_value).parameters)
    expected = set(getattr(attribute_value, "__annotations__", {}))
    extra = actual - expected
    if extra and extra != {"self"}:
        raise ValueError(f"Got unexpected fields: {extra} in {attribute_name}")
    missing = expected - actual
    if missing and missing != {"return"}:
        raise ValueError(f"Missing required fields: {missing} in {attribute_name}")


class AutopycoinMetaClass(abc.ABCMeta):
    """Metaclass for autopycoin objects."""

    def __init__(cls, name, bases, namespace):
        if not namespace.get("_ap_do_not_transform_this_class", False):
            for attribute_name, attribute_value in _methods_to_inspect(cls):
                _check_field_annotations(attribute_name, attribute_value)
                _wrap_user_constructor(cls, attribute_name, attribute_value)
        super().__init__(name, bases, namespace)


class AutopycoinMetaLayer(AutopycoinMetaClass):
    """Metaclass for autopycoin layers."""

    def __init__(cls, name, bases, namespace):
        if not namespace.get("_ap_do_not_transform_this_class", False):
            cls.build = _wrap_build(cls.build)
            cls.call = _wrap_call(cls.call)

        super().__init__(name, bases, namespace)


def _wrap_build(fn):
    """Wrap the build method with a init_params function"""

    def build_wrapper(self, inputs_shape):
        self.init_params(inputs_shape)
        return fn(self, inputs_shape)

    return build_wrapper


def _wrap_call(fn):
    """Wrap the call method with a _preprocessing and post_processing methods"""

    def call_wrapper(self, inputs, *args, **kwargs):
        inputs = self._preprocessing_wrapper(inputs)
        outputs = fn(self, inputs, *args, **kwargs)
        outputs = self._post_processing_wrapper(outputs)
        return outputs

    return call_wrapper


# ==============================================================================
# Base class for autopycoin objects
# ==============================================================================
class AutopycoinBase:
    """
    A new autopycoin class has to inherit from this base class.
    It checks type and lenght of arguments for each methods inside a class.
    `__validate__` let user defines its own validation attributes.
    """

    _ap_do_not_transform_this_class = True
    NOT_INSPECT = []

    @classmethod
    def _type_fields(
        cls, attribute_name, attribute_value
    ):  # pylint: disable=no-self-argument
        """An ordered list describing the fields of this cls.
        Returns
        -------
        A list of `AutopycoinField` objects.  Forward references are resolved
        if possible, or left unresolved otherwise.
        """
        if "_ap_type_cached_fields" in cls.__dict__:  # do not inherit.
            if f"{cls}_{attribute_name}" in cls._ap_type_cached_fields:
                return cls._ap_type_cached_fields
            list_fields = cls._ap_type_cached_fields
        else:
            list_fields = {}

        try:
            type_hints = typing.get_type_hints(attribute_value)
            ok_to_cache = True  # all forward references have been resolved.
        except (NameError, AttributeError):
            # Unresolved forward reference -- gather type hints manually.
            # * NameError comes from an annotation like `Foo` where class
            #   `Foo` hasn't been defined yet.
            # * AttributeError comes from an annotation like `foo.Bar`, where
            #   the module `foo` exists but `Bar` hasn't been defined yet.
            # Note: If a user attempts to instantiate a `ExtensionType` type that
            # still has unresolved forward references (e.g., because of a typo or a
            # missing import), then the constructor will raise an exception.
            type_hints = {}
            for base in reversed(cls.__mro__):
                type_hints.update(base.__dict__.get("__annotations__", {}))
            ok_to_cache = False
        fields = []
        signatures = signature(attribute_value)  # To get default values
        _pop_field(type_hints)  # pop not supported fields
        for (name, value_type) in type_hints.items():
            default = signatures.parameters[name].default
            if callable(default):
                try:
                    default = default()
                except TypeError:
                    pass
                if isinstance(default, _empty):
                    default = AutopycoinField.NO_DEFAULT
            fields.append(
                AutopycoinField(
                    name, value_type, default
                )  # pylint: disable=too-many-function-args
            )

        list_fields.update({f"{cls}_{attribute_name}": fields})

        if ok_to_cache:
            cls._ap_type_cached_fields = list_fields

        return list_fields

    def __validate__(self, cls, output, method_name, *args, **kwargs):
        getattr(cls, "_val_" + method_name, dummy_callable)(
            self, output, *args, **kwargs
        )

    def _get_parameter(self, args: list, kwargs: dict, name: str, position: int):
        param = kwargs.get(name, None)
        if param is None:
            param = args[position]
        return param


class AutopycoinBaseClass(AutopycoinBase, metaclass=AutopycoinMetaClass):
    _ap_do_not_transform_this_class = True


class AutopycoinBaseLayer(AutopycoinBase, metaclass=AutopycoinMetaLayer):
    _ap_do_not_transform_this_class = True
