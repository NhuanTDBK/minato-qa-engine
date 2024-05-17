from tempfile import TemporaryDirectory
import json
import os
import unittest
from src.writer.json import JsonWriter as JSONWriter


class TestJSONWriter(unittest.TestCase):
    def test_write_rotate_file(self):
        tmp_dir = TemporaryDirectory()
        writer = JSONWriter(tmp_dir.name, rotate=2)
        with writer:
            writer.write({"a": 1})
            writer.write({"b": 2})
            writer.write({"c": 3})
            writer.write({"d": 4})
            writer.write({"e": 5})

        assert len(os.listdir(tmp_dir.name)) == 3
        with open(os.path.join(tmp_dir.name, "file_0.json")) as f:
            assert json.load(f) == [{"a": 1}, {"b": 2}]
        with open(os.path.join(tmp_dir.name, "file_1.json")) as f:
            assert json.load(f) == [{"c": 3}, {"d": 4}]
        with open(os.path.join(tmp_dir.name, "file_2.json")) as f:
            assert json.load(f) == [{"e": 5}]


if __name__ == "__main__":
    unittest.main()
