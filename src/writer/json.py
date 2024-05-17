import os
import json


# Rotated append to json file, after N rows, create a new file


class JsonWriter:
    def __init__(self, folder_path, rotate=1000):
        self.folder_path = folder_path
        self.rotate = rotate
        self.file = None
        self.count = 0
        self.total_count = 0
        self.idx = 0

    def get_file_path(self):
        return os.path.join(self.folder_path, f"file_{self.idx}.json")

    def __enter__(self):
        self.file = open(self.get_file_path(), "a+")
        self.file.write("[\n")
        return self

    def write(self, data):
        if self.count % self.rotate == 0 and self.count > 0:
            self.file.write("]")
            self.file.close()
            self.idx += 1
            self.file = open(self.get_file_path(), "a+")
            self.file.write("[")
            self.count = 0
        if self.count > 0:
            self.file.write(",")
        json.dump(data, self.file, indent=4)
        self.count += 1
        self.total_count += 1

    def flush(self):
        self.file.write("]")
        self.file.close()
        self.file = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        return False
