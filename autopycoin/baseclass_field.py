"""Metadata about fields for autopycoin classes.
Most of the functions have been taken from tensorflow nightly."""

import enum
import collections
import typing
import pandas as pd
import numpy as np
import tensorflow as tf

# Note: typing.get_args was added in Python 3.8.
if hasattr(typing, "get_args"):
    get_generic_type_args = typing.get_args
else:
    get_generic_type_args = lambda tp: tp.__args__


class _ConversionContext(enum.Enum):
    """
    Enum to indicate what kind of value is being converted.
    Used by `_convert_value` and their helper methods.
    """

    VALUE = 1
    DEFAULT = 2
    NO_DEFAULT = 3
    TUPLE = 4
    UNION = 5
    MAPPING = 6


class Sentinel(object):
    """Sentinel value that's not equal (w/ `is`) to any user value."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name


class AutopycoinField(
    collections.namedtuple("AutopycoinField", ["name", "value_type", "default"])
):
    """Metadata about a single field in an autopycoin object."""

    NO_DEFAULT = Sentinel("AutopycoinField.NO_DEFAULT")

    def __new__(cls, name, value_type, default=NO_DEFAULT):
        """Constructs a new Autopycoinfield containing metadata for a single field.

        Parameters
        ----------
        name: The name of the new field (`str`).
        value_type: A python type expression constraining what values this field
            can take.
        default: The default value for the new field, or `NO_DEFAULT` if this
            field has no default value.
        Returns
        -------
          A new `AutopycoinField`.
        """

        try:
            validate_field_value_type(value_type, allow_forward_references=True)
        except TypeError as error:
            raise TypeError(f"In field {name!r}: {error}") from error

        if not isinstance(default, Sentinel):
            default = _convert_value(
                default,
                value_type,
                (f"default value for {name}",),
                _ConversionContext.DEFAULT,
            )

        return super().__new__(cls, name, value_type, default)


def validate_field_value_type(
    value_type, in_mapping_key=False, allow_forward_references=False
):
    """Checks that `value_type` contains only supported type annotations.
    Args:
        value_type: The type annotation to check.
        in_mapping_key: True if `value_type` is nested in the key of a mapping.
        allow_forward_references: If false, then raise an exception if a
        `value_type` contains a forward reference (i.e., a string literal).
    Raises:
        TypeError: If `value_type` contains an unsupported type annotation.
    """
    if isinstance(value_type, str) or is_forward_ref(value_type):
        if allow_forward_references:
            return
        else:
            raise TypeError(f"Unresolved forward reference {value_type!r}")

    if value_type in (
        int,
        float,
        str,
        bytes,
        bool,
        None,
        pd.DataFrame,
        np.array,
        tf.Tensor,
        typing.Callable,
    ):
        return
    elif (
        is_generic_tuple(value_type)
        or is_generic_union(value_type)
        or is_generic_list(value_type)
    ):
        type_args = get_generic_type_args(value_type)
        if (
            len(type_args) == 2
            and type_args[1] is Ellipsis
            and is_generic_tuple(value_type)
        ):  # `Tuple[X, ...]`
            validate_field_value_type(
                type_args[0], in_mapping_key, allow_forward_references
            )
        else:
            for arg in get_generic_type_args(value_type):
                validate_field_value_type(arg, in_mapping_key, allow_forward_references)
    elif is_generic_mapping(value_type):
        key_type, value_type = get_generic_type_args(value_type)
        validate_field_value_type(key_type, True, allow_forward_references)
        validate_field_value_type(value_type, in_mapping_key, allow_forward_references)
    elif isinstance(value_type, type):
        return
    else:
        raise TypeError(f"Unsupported type annotation {value_type!r}")


def _convert_value(
    value, expected_type, path, context=_ConversionContext.VALUE, convert=True
):
    """Type-checks value.

    Parameters
    ----------
    value: The value to type-check.
    expected_type: The expected type for the value.
    path: Tuple of `str` naming the value (used for exception messages).
    context: _ConversionContext, indicates what kind of value we are converting.

    Raises
    ------
    TypeError: If `value` can not be converted to the expected type.
    """
    assert isinstance(path, tuple)
    if is_generic_tuple(expected_type):
        return _check_tuple(value, expected_type, path, context)
    elif is_generic_mapping(expected_type):
        return _check_mapping(value, expected_type, path, context)
    elif is_generic_union(expected_type):
        return _check_union(value, expected_type, path, context)
    elif is_generic_list(expected_type):
        return _check_list(value, expected_type, path, context)
    elif isinstance(value, expected_type):
        return value

    raise TypeError(
        f'{"".join(path)}: expected {expected_type.__name__}, got {value!r}'
    )


def _check_tuple(value, expected_type, path, context):
    """Checks `value` to a tuple with type `expected_type`."""
    if not isinstance(value, typing.Sequence):
        raise TypeError(f'{"".join(path)}: expected tuple, got {value!r}')
    element_types = get_generic_type_args(expected_type)
    if len(element_types) == 2 and element_types[1] is Ellipsis:
        return tuple(
            [
                _convert_value(
                    v, element_types[0], path + (f"[{i}]",), context, convert=False
                )
                for (i, v) in enumerate(value)
            ]
        )
    else:
        if len(value) != len(element_types):
            raise TypeError(
                f'{"".join(path)}: expected tuple with length '
                f"{len(element_types)}, got {value!r})"
            )
        return tuple(
            [
                _convert_value(v, t, path + (f"[{i}]",), context, convert=False)
                for (i, (v, t)) in enumerate(zip(value, element_types))
            ]
        )


def _check_mapping(value, expected_type, path, context):
    """Checks `value` to a mapping with type `expected_type`."""
    if not isinstance(value, typing.Mapping):
        raise TypeError(f'{"".join(path)}: expected mapping, got {value!r}')
    key_type, value_type = get_generic_type_args(expected_type)
    return dict(
        [
            (
                _convert_value(k, key_type, path + ("[<key>]",), context, convert=True),
                _convert_value(
                    v, value_type, path + (f"[{k!r}]",), context, convert=True
                ),
            )
            for (k, v) in value.items()
        ]
    )


def _check_union(value, expected_type, path, context):
    """Checks `value` to a value with any of the types in `expected_type`."""
    for type_option in get_generic_type_args(expected_type):
        try:
            return _convert_value(value, type_option, path, context, convert=False)
        except TypeError:
            pass
    return _convert_value(
        value, get_generic_type_args(expected_type)[-1], path, context, convert=True
    )


def _check_list(value, expected_type, path, context):
    """Checks `value` to a value with any of the types in `expected_type`."""
    if not isinstance(value, typing.Sequence):
        raise TypeError(f'{"".join(path)}: expected list, got {value!r}')
    element_type = get_generic_type_args(expected_type)[0]
    return [
        _convert_value(v, element_type, path + (f"[{i}]",), context, convert=True)
        for (i, v) in enumerate(value)
    ]


def is_generic_union(tp):
    """Returns true if `tp` is a parameterized typing.Union value."""
    return tp is not typing.Union and getattr(tp, "__origin__", None) is typing.Union


def is_generic_tuple(tp):
    """Returns true if `tp` is a parameterized typing.Tuple value."""
    return tp not in (tuple, typing.Tuple) and getattr(tp, "__origin__", None) in (
        tuple,
        typing.Tuple,
    )


def is_generic_list(tp):
    """Returns true if `tp` is a parameterized typing.List value."""
    return tp not in (list, typing.List) and getattr(tp, "__origin__", None) in (
        list,
        typing.List,
    )


def is_generic_mapping(tp):
    """Returns true if `tp` is a parameterized typing.Mapping value."""
    return tp not in (collections.abc.Mapping, typing.Mapping) and getattr(
        tp, "__origin__", None
    ) in (collections.abc.Mapping, typing.Mapping)


def is_forward_ref(tp):
    """Returns true if `tp` is a typing forward reference."""
    if hasattr(typing, "ForwardRef"):
        return isinstance(tp, typing.ForwardRef)
    elif hasattr(typing, "_ForwardRef"):
        return isinstance(tp, typing._ForwardRef)  # pylint: disable=protected-access
    else:
        return False
