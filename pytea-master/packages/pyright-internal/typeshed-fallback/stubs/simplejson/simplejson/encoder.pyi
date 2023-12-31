from typing import Any

class JSONEncoder(object):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def encode(self, o: Any): ...
    def default(self, o: Any): ...
    def iterencode(self, o: Any, _one_shot: bool): ...

class JSONEncoderForHTML(JSONEncoder): ...
