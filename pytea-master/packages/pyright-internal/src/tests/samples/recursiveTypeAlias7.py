# This sample tests a recursive type alias used within
# a recursive function.

from typing import Dict, Union


A = Union[str, Dict[str, "A"]]


def foo(x: A):
    if isinstance(x, str):
        print(x)
    else:
        for _, v in x.items():
            foo(v)
