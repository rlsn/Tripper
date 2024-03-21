"""
An attribute dictionary that's very handy everywhere
"""
import json
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def to_json(self, filename):
        with open(filename, "w") as wf:
            wf.write(json.dumps(self, indent=4))

    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as f:
            obj = cls(json.loads(f.read()))
        return obj