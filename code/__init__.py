"""Project package with the public API of Python's homonymous stdlib module."""

import importlib.util
import sysconfig
from pathlib import Path


_stdlib_path = Path(sysconfig.get_path("stdlib")) / "code.py"
_spec = importlib.util.spec_from_file_location("_python_stdlib_code", _stdlib_path)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load Python's standard-library code module")
_stdlib_code = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stdlib_code)

InteractiveInterpreter = _stdlib_code.InteractiveInterpreter
InteractiveConsole = _stdlib_code.InteractiveConsole
interact = _stdlib_code.interact
compile_command = _stdlib_code.compile_command

__all__ = [
    "InteractiveInterpreter",
    "InteractiveConsole",
    "interact",
    "compile_command",
]
