"""
Defines assertion function
"""

from typing import Any, List, Union

from .utils import convert_to_list


def is_between(
    obj_to_compare: Any,
    lower_limit: Union[int, float],
    upper_limit: Union[int, float],
    name: str,
) -> None:
    """
    Assert that `obj_to_compare` is between `lower_limit` and `upper_limit`,
    If a list is provided, the comparison is performed over all elements.

    Parameters
    ----------
    obj_to_compare: Any
        object to compare to `target`.
    lower_limit: int | float
        The lower treshold.
    upper_limit: int | float
        The upper treshold.
    name: str
        Name to print in case of failure.

    raises
    ------
    AssertionError
        If `obj_to_compare` or its elements are not between `lower_limit` and `upper_limit`.
    """
    obj_to_compare = convert_to_list(obj_to_compare)
    assert all(
        value >= lower_limit and value <= upper_limit for value in obj_to_compare
    ), f"""`{name}` or its elements has to be between {lower_limit} and {upper_limit}.
            Got {obj_to_compare}."""


def greater_or_equal(obj_to_compare: Any, target: Union[int, float], name: str) -> None:
    """
    Assert that `obj_to_compare` is greater or equal to `target`,
    If a list is provided, the comparison is performed over all elements.

    Parameters
    ----------
    obj_to_compare: Any
        object to compare to `target`.
    target: int | float
        The treshold.
    name: str
        Name to print in case of failure.

    raises
    ------
    AssertionError
        If `obj_to_compare` or its elements are not greater or equals to target.
    """
    obj_to_compare = convert_to_list(obj_to_compare)
    assert all(
        value >= target for value in obj_to_compare
    ), f"""`{name}` or its elements has to be greater or equal than {target}.
            Got {obj_to_compare}."""


def lower_or_equal(obj_to_compare: Any, target: Union[int, float], name: str) -> None:
    """
    Assert that `obj_to_compare` is lower or equal to `target`,
    If a list is provided, the comparison is performed over all elements.

    Parameters
    ----------
    obj_to_compare: Any
        object to compare to `target`.
    target: int | float
        The treshold.
    name: str
        Name to print in case of failure.

    raises
    ------
    AssertionError
        If `obj_to_compare` or its elements are not lower or equals to target.
    """
    obj_to_compare = convert_to_list(obj_to_compare)
    assert all(
        value <= target for value in obj_to_compare
    ), f"""`{name}` or its elements has to be lower or equal than {target}.
            Got {obj_to_compare}."""


def equal_length(
    list_one: List[Union[int, float]],
    list_two: List[Union[int, float]],
    name_one: str,
    name_two: str,
):
    """
    Assert that the two lists have the same length.

    Parameters
    ----------
    list_one: list[int | float]
        First list to compare.
    list_two: list[int | float]
        Second list to compare.
    name_one: str
        Name to print in case of failure.
    name_two: str
        Name to print in case of failure.

    raises
    ------
    AssertionError
        If the two lists doesn't have the same length.
    """
    len_one = len(convert_to_list(list_one))
    len_two = len(convert_to_list(list_two))
    assert len_one == len_two, (
        f"`{name_one}` and `{name_two}` are expected "
        f"to have the same length, got "
        f"{len_one} and {len_two} respectively."
    )
