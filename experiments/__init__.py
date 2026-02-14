import importlib, inspect
from pathlib import Path
import sys


"""
module_path: parent directory path with . as subfile dividers
f: filename to import modules from
"""


def import_modules_from_files(module_path, f):
    if f[0] != "_" and f[0] != ".":
        module_name = module_path + "." + f.split(".")[0]
        m = importlib.import_module(module_name)
        print("imported", module_name)
        for name, obj in inspect.getmembers(m):
            if inspect.isclass(obj):
                setattr(thismodule, name, getattr(m, name))
        del m
        del name
        del obj


path = Path(__path__[0])
thismodule = sys.modules[__name__]
print(thismodule)
print(path)
files = [f.name for f in path.iterdir() if not f.name.endswith(".ini")]
module_path = __name__
# Import general first so that it can be a superclass
fpath = path / "general"
subfiles = [f.name for f in fpath.iterdir() if not f.name.endswith(".ini")]
submodule_path = module_path + ".general"
for subf in subfiles:
    import_modules_from_files(submodule_path, subf)

for f in files:
    fpath = path / f
    print(fpath)
    if f[0] == "_" or f[0] == "." or f == "general":
        continue
    if fpath.is_dir():
        subfiles = [sf.name for sf in fpath.iterdir() if not sf.name.endswith(".ini")]
        submodule_path = module_path + "." + fpath.name
        for subf in subfiles:
            import_modules_from_files(submodule_path, subf)
    else:
        import_modules_from_files(module_path, f)

del f
del thismodule
del files
