from collections import OrderedDict
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from .models import MODEL_NAMES, canonical_model_name


class ConfigError(ValueError):
    pass


class StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML field names must be strings")
        if key in mapping:
            raise ConfigError("Duplicate YAML key: {}".format(key))
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

BENCHMARK_FIELDS = {
    "data_root": "path",
    "data_config": "path",
    "seed": "nonnegative_int",
    "seeds": "nonnegative_int_list",
    "epochs": "positive_int",
    "batch_size": "positive_int",
    "wavelet_epochs": "positive_int",
    "wavelet_batch_size": "positive_int",
    "learning_rate": "positive_number",
    "crop_length": "positive_int",
    "num_workers": "nonnegative_int",
    "device": "string",
    "official_raw_data": "bool",
    "cache_dir": "path",
    "resume": "bool",
    "no_mixed_precision": "bool",
    "noisy_manifest": "path",
    "noisy_root": "path",
    "denoised_manifest": "path",
    "denoised_root": "path",
}

GLOBAL_FIELDS = dict(BENCHMARK_FIELDS, fail_fast="bool")

TASK_OPTION_FIELDS = {
    "prepare_data": {
        "archive": "path_list",
        "search_root": "path_list",
        "workspace": "path",
        "output_dir": "path",
    },
    "train": BENCHMARK_FIELDS,
    "evaluate": BENCHMARK_FIELDS,
    "report": {
        "input_root": "path",
        "output_dir": "path",
        "expected_seeds": "nonnegative_int_list",
        "excluded_wavelet_status": "string",
    },
    "package": {
        "input_root": "path",
        "output_file": "path",
    },
}

def _mapping(value, location):
    if not isinstance(value, dict):
        raise ConfigError("{} must be a mapping".format(location))
    return value


def _check_keys(value, allowed, location, required=()):
    if not all(isinstance(key, str) for key in value):
        raise ConfigError("{} field names must be strings".format(location))
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ConfigError("{} has unknown field(s): {}".format(location, ", ".join(unknown)))
    missing = sorted(set(required) - set(value))
    if missing:
        raise ConfigError("{} is missing field(s): {}".format(location, ", ".join(missing)))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_value(value, kind, location):
    if kind == "bool" and not isinstance(value, bool):
        raise ConfigError("{} must be a boolean".format(location))
    if kind == "string" and (not isinstance(value, str) or not value.strip()):
        raise ConfigError("{} must be a non-empty string".format(location))
    if kind == "path" and (not isinstance(value, str) or not value.strip()):
        raise ConfigError("{} must be a non-empty path string".format(location))
    if kind == "int" and not _is_int(value):
        raise ConfigError("{} must be an integer".format(location))
    if kind == "positive_int" and (not _is_int(value) or value < 1):
        raise ConfigError("{} must be a positive integer".format(location))
    if kind == "nonnegative_int" and (not _is_int(value) or value < 0):
        raise ConfigError("{} must be a non-negative integer".format(location))
    if kind == "positive_number" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ConfigError("{} must be a positive number".format(location))
    if kind == "int_list":
        if not isinstance(value, list) or not value or not all(_is_int(item) for item in value):
            raise ConfigError("{} must be a non-empty integer list".format(location))
    if kind == "nonnegative_int_list":
        if not isinstance(value, list) or not value or not all(
                _is_int(item) and item >= 0 for item in value):
            raise ConfigError("{} must be a non-empty non-negative integer list".format(location))
    if kind == "path_list":
        if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value):
            raise ConfigError("{} must be a path string list".format(location))


def _resolve_path(value, base):
    path = Path(value).expanduser()
    if not path.is_absolute() and (
            PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()):
        return value
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _validate_options(value, fields, location, base):
    value = _mapping(value, location)
    _check_keys(value, fields, location)
    result = {}
    for key, item in value.items():
        kind = fields[key]
        _validate_value(item, kind, "{}.{}".format(location, key))
        if kind == "path":
            item = _resolve_path(item, base)
        elif kind == "path_list":
            item = [_resolve_path(path, base) for path in item]
        result[key] = item
    if "seed" in result and "seeds" in result:
        raise ConfigError("{} cannot contain both seed and seeds".format(location))
    return result


def _normalize_model(name, location):
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("{} must be a non-empty model name".format(location))
    canonical = canonical_model_name(name)
    if canonical not in MODEL_NAMES:
        raise ConfigError("{} has unknown model {!r}".format(location, name))
    return canonical


def _load_models(value, base):
    if value is None:
        return OrderedDict((name, {}) for name in MODEL_NAMES)
    result = OrderedDict()
    if isinstance(value, list):
        entries = []
        for index, item in enumerate(value):
            location = "models[{}]".format(index)
            if isinstance(item, str):
                entries.append((item, {}, location))
            else:
                item = _mapping(item, location)
                _check_keys(item, {"name", "overrides"}, location, {"name"})
                overrides = item.get("overrides", {})
                entries.append((item["name"], {} if overrides is None else overrides, location))
    elif isinstance(value, dict):
        entries = [(name, {} if overrides is None else overrides, "models.{}".format(name))
                   for name, overrides in value.items()]
    else:
        raise ConfigError("models must be a list or mapping")
    for name, overrides, location in entries:
        canonical = _normalize_model(name, location)
        if canonical in result:
            raise ConfigError("models contains duplicate model {!r}".format(canonical))
        result[canonical] = _validate_options(
            overrides, BENCHMARK_FIELDS, "{}.overrides".format(location), base)
    if not result:
        raise ConfigError("models cannot be empty")
    return result


def _load_groups(value):
    if value is None:
        return {}
    value = _mapping(value, "model_groups")
    result = {}
    for name, members in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("model_groups keys must be non-empty strings")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ConfigError("model group names may only contain letters, digits, '.', '_' and '-'")
        if canonical_model_name(name) in MODEL_NAMES:
            raise ConfigError("model group {!r} conflicts with a model name".format(name))
        if not isinstance(members, list) or not members or not all(
                isinstance(member, str) and member.strip() for member in members):
            raise ConfigError("model_groups.{} must be a non-empty string list".format(name))
        result[name] = members
    return result


def _expand_group(name, groups, stack):
    reference = name[1:] if name.startswith("@") else name
    canonical = canonical_model_name(reference)
    if canonical in MODEL_NAMES:
        return [canonical]
    if reference not in groups:
        raise ConfigError("Unknown model or model group {!r}".format(name))
    if reference in stack:
        cycle = stack[stack.index(reference):] + [reference]
        raise ConfigError("Model group cycle detected: {}".format(" -> ".join(cycle)))
    models = []
    for member in groups[reference]:
        for model in _expand_group(member, groups, stack + [reference]):
            if model not in models:
                models.append(model)
    return models


def _load_tasks(value, groups, default_models, base):
    if not isinstance(value, list) or not value:
        raise ConfigError("tasks must be a non-empty list")
    tasks = []
    names = set()
    for index, item in enumerate(value):
        location = "tasks[{}]".format(index)
        item = _mapping(item, location)
        _check_keys(item, {"name", "id", "type", "depends_on", "models", "options"},
                    location, {"type"})
        if ("name" in item) == ("id" in item):
            raise ConfigError("{} must contain exactly one of name or id".format(location))
        name = item.get("name", item.get("id"))
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("{}.name must be a non-empty string".format(location))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ConfigError("{}.name may only contain letters, digits, '.', '_' and '-'".format(
                location))
        if name in names:
            raise ConfigError("Duplicate task name {!r}".format(name))
        names.add(name)
        task_type = item["type"]
        if task_type not in TASK_OPTION_FIELDS:
            raise ConfigError("{}.type must be one of {}".format(
                location, ", ".join(TASK_OPTION_FIELDS)))
        depends_on = item.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
                isinstance(dependency, str) and dependency.strip() for dependency in depends_on):
            raise ConfigError("{}.depends_on must be a string list".format(location))
        if len(depends_on) != len(set(depends_on)):
            raise ConfigError("{}.depends_on contains duplicates".format(location))
        selected = item.get("models", list(default_models))
        if isinstance(selected, str):
            selected = [selected]
        if not isinstance(selected, list) or not selected or not all(
                isinstance(model, str) and model.strip() for model in selected):
            raise ConfigError("{}.models must be a model/group name or non-empty list".format(location))
        expanded = []
        for reference in selected:
            for model in _expand_group(reference, groups, []):
                if model not in expanded:
                    expanded.append(model)
        if task_type not in ("train", "evaluate") and "models" in item:
            raise ConfigError("{}.models is only valid for train and evaluate tasks".format(location))
        options = _validate_options(item.get("options", {}), TASK_OPTION_FIELDS[task_type],
                                    "{}.options".format(location), base)
        if task_type == "prepare_data" and not (
                options.get("archive") or options.get("search_root")):
            raise ConfigError(
                "{}.options requires at least one archive or search_root".format(location))
        tasks.append({
            "name": name,
            "type": task_type,
            "depends_on": depends_on,
            "models": expanded if task_type in ("train", "evaluate") else [],
            "options": options,
        })
    return tasks


def topological_tasks(tasks):
    by_name = {task["name"]: task for task in tasks}
    for task in tasks:
        unknown = [name for name in task["depends_on"] if name not in by_name]
        if unknown:
            raise ConfigError("Task {!r} has unknown dependencies: {}".format(
                task["name"], ", ".join(unknown)))
    state = {}
    ordered = []

    def visit(name, stack):
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            cycle = stack[stack.index(name):] + [name]
            raise ConfigError("Task dependency cycle detected: {}".format(" -> ".join(cycle)))
        state[name] = 1
        for dependency in by_name[name]["depends_on"]:
            visit(dependency, stack + [name])
        state[name] = 2
        ordered.append(by_name[name])

    for task in tasks:
        visit(task["name"], [])
    return ordered


def load_config(path):
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.load(stream, Loader=StrictLoader)
    except OSError as error:
        raise ConfigError("Cannot read config {}: {}".format(config_path, error))
    except yaml.YAMLError as error:
        raise ConfigError("Invalid YAML in {}: {}".format(config_path, error))
    loaded = _mapping(loaded, "YAML root")
    _check_keys(loaded, {"version", "output_dir", "global", "models", "model_groups", "tasks"},
                "YAML root", {"output_dir", "tasks"})
    version = loaded.get("version", 1)
    if not _is_int(version) or version != 1:
        raise ConfigError("version must be 1")
    base = config_path.parent
    _validate_value(loaded["output_dir"], "path", "output_dir")
    output_dir = _resolve_path(loaded["output_dir"], base)
    global_options = _validate_options(
        loaded.get("global", {}), GLOBAL_FIELDS, "global", base)
    models = _load_models(loaded.get("models"), base)
    groups = _load_groups(loaded.get("model_groups"))
    for group in groups:
        _expand_group(group, groups, [])
    tasks = _load_tasks(loaded["tasks"], groups, models, base)
    tasks = topological_tasks(tasks)
    return {
        "version": 1,
        "config_path": str(config_path),
        "output_dir": output_dir,
        "global": global_options,
        "models": dict(models),
        "model_groups": {name: _expand_group(name, groups, []) for name in groups},
        "tasks": tasks,
    }
