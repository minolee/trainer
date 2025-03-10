import json
import csv
# import ruamel.yaml as yaml
import yaml
import os
from .func_util import create_register_deco

__all__ = ["iter_dir", "read_txt", "read_jsonl", "read_csv", "read_tsv", "read_yaml", "read_magic",
              "write_txt", "write_jsonl", "write_csv", "write_tsv", "write_yaml", "write_magic"]

_reader_fn: dict = {}
_writer_fn: dict = {}

reader = create_register_deco(_reader_fn)
writer = create_register_deco(_writer_fn)
# def reader(fn):
#     global _reader_fn
#     _reader_fn[fn.__name__] = fn
#     def decorator(*args, **kwargs):
#         return fn(*args, **kwargs)
#     return decorator

# def writer(fn):
#     global _writer_fn
#     _writer_fn[fn.__name__] = fn
#     def decorator(*args, **kwargs):
#         return fn(*args, **kwargs)
#     return decorator

def iter_dir(p, *, prefix=None, filter_prefix=None, postfix=None, filter_postfix=None, return_absolute=False):
    for p, d, fs in os.walk(p):
        for f in fs:
            if prefix and not f.startswith(prefix): continue
            if filter_prefix and f.startswith(filter_prefix): continue
            if postfix and not f.endswith(postfix): continue
            if filter_postfix and f.endswith(filter_postfix): continue
            path = os.path.join(p, f)
            if not return_absolute: path = path.replace(p, "").lstrip("/")
            yield path
            

@reader
def read_txt(f):
    with open(f, "r") as f:
        for line in f:
            yield line.strip()
@reader
def read_jsonl(f):
    with open(f, "r") as f:
        for line in f:
            yield json.loads(line)

@reader
def read_csv(f, delimiter=","):
    with open(f, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            yield row
@reader
def read_tsv(f):
    return read_csv(f, delimiter="\t")

@reader
def read_json(f):
    with open(f, encoding="UTF8") as f:
        return json.load(f)

@reader
def read_yaml(f):
    with open(f, "r") as f:
        return yaml.safe_load(f)

def read_magic(f, ext=None):
    """File extension에 따라서 reader를 선택하여 읽어옴"""
    ext = ext or f.split(".")[-1]
    return _reader_fn[f"read_{ext}"](f)

@writer
def write_txt(f, data):
    with open(f, "w", encoding="UTF8") as f:
        for line in data:
            f.write(line + "\n")

@writer
def write_jsonl(f, data):
    with open(f, "w", encoding="UTF8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

@writer
def write_csv(f, data, delimiter=","):
    with open(f, "w", encoding="UTF8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        for row in data:
            writer.writerow(row)

@writer
def write_tsv(f, data):
    return write_csv(f, data, delimiter="\t")

@writer
def write_yaml(f, data):
    with open(f, "w", encoding="UTF8") as f:
        yaml.dump(data, f)

def write_magic(f, data, ext=None):
    """File extension에 따라서 writer를 선택하여 쓰기"""
    ext = ext or f.split(".")[-1]
    return _writer_fn[f"write_{ext}"](f, data)