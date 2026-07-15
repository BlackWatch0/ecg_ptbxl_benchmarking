import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


SNRS = (24, 12, 6, 0, -6)
ID_COLUMNS = ("ecg_id", "recordnumber", "record_number", "record_id")
PATH_COLUMNS = ("wfdb_record_relative", "final_wfdb_record_relative", "record_path",
                "wfdb_path", "filename_lr", "filename", "recordfile", "record_file", "path")
SNR_COLUMNS = ("snr_target_db", "snr_db", "target_snr_db", "snr")
CONDITION_COLUMNS = ("condition", "signal_type", "variant", "dataset", "scenario", "data_type")
FOLD_COLUMNS = ("strat_fold", "fold")


def _safe_destination(root, name):
    name = name.replace("\\", "/")
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("Archive contains unsafe path: {}".format(name))
    destination = (root / Path(*member.parts)).resolve()
    root = root.resolve()
    if destination != root and root not in destination.parents:
        raise ValueError("Archive contains unsafe path: {}".format(name))
    return destination


def inspect_and_extract(archive, workspace):
    archive = Path(archive).resolve()
    if not archive.is_file():
        raise FileNotFoundError("Archive does not exist: {}".format(archive))
    fingerprint = hashlib.sha256(str(archive).encode("utf-8")).hexdigest()[:12]
    destination = Path(workspace) / "{}_{}".format(archive.stem, fingerprint)
    marker = destination / ".extraction_complete.json"
    identity = {"archive": str(archive), "size": archive.stat().st_size,
                "mtime_ns": archive.stat().st_mtime_ns}
    if marker.is_file() and json.loads(marker.read_text()) == identity:
        print("Reusing extracted archive: {}".format(destination))
        return destination
    if destination.exists():
        raise FileExistsError("Refusing to replace incomplete or changed extraction: {}".format(destination))
    destination.mkdir(parents=True)
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                members = source.infolist()
                for member in members:
                    _safe_destination(destination, member.filename)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise ValueError("Archive contains a symbolic link: {}".format(member.filename))
                print("Inspected {} ZIP members in {}".format(len(members), archive))
                source.extractall(destination)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as source:
                members = source.getmembers()
                for member in members:
                    _safe_destination(destination, member.name)
                    if member.issym() or member.islnk() or member.isdev():
                        raise ValueError("Archive contains an unsafe special entry: {}".format(member.name))
                print("Inspected {} tar members in {}".format(len(members), archive))
                source.extractall(destination)
        else:
            raise ValueError("Unsupported archive format: {}".format(archive))
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    print("Extracted archive: {}".format(destination))
    return destination


def _columns(fieldnames):
    return {name.strip().lower(): name for name in (fieldnames or [])}


def _column(columns, choices):
    return next((columns[name] for name in choices if name in columns), None)


def _integer(value):
    return int(float(str(value).strip()))


def _snr_from_text(value):
    text = str(value).lower().replace("minus", "m")
    match = re.search(r"snr_?(m?-?\d+)", text)
    if not match:
        match = re.search(r"(?<!\d)(-?\d+)\s*db", text)
    if not match:
        return None
    token = match.group(1)
    return -int(token[1:]) if token.startswith("m") else int(token)


def _condition(value):
    text = str(value).lower()
    if "denois" in text:
        return "denoised"
    if "clean" in text or "no_noise" in text:
        return "clean"
    if "nois" in text or "mixed" in text or "snr" in text:
        return "noisy"
    return "clean"


def _record_stem(value):
    value = str(value).strip().replace("\\", "/")
    for suffix in (".hea", ".dat", ".mat"):
        if value.lower().endswith(suffix):
            return value[:-len(suffix)]
    return value


def _waveform_index(roots):
    by_name = {}
    for root in roots:
        for header in root.rglob("*.hea"):
            by_name.setdefault(header.stem, []).append(header.with_suffix(""))
    return by_name


def _resolve_record(value, csv_path, roots, index):
    stem = _record_stem(value)
    candidates = [csv_path.parent / stem]
    candidates.extend(root / stem for root in roots)
    for candidate in candidates:
        if candidate.with_suffix(".hea").is_file() and candidate.with_suffix(".dat").is_file():
            return candidate.resolve()
    matches = [path for path in index.get(Path(stem).name, [])
               if path.with_suffix(".dat").is_file()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        suffix = stem.strip("/")
        suffix_matches = [path for path in matches if path.as_posix().endswith(suffix)]
        if len(suffix_matches) == 1:
            return suffix_matches[0].resolve()
    raise FileNotFoundError("Cannot uniquely resolve WFDB record {!r} from {}".format(value, csv_path))


def _csv_layout_from_fields(fieldnames):
    columns = _columns(fieldnames)
    id_column = _column(columns, ID_COLUMNS)
    path_column = _column(columns, PATH_COLUMNS)
    if not id_column or not path_column:
        return None
    return columns, id_column, path_column


def _csv_layout(path):
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            return _csv_layout_from_fields(csv.DictReader(source).fieldnames)
    except UnicodeDecodeError:
        return None


def discover_and_validate(roots, output_dir):
    roots = [Path(root).resolve() for root in roots]
    missing_roots = [root for root in roots if not root.exists()]
    if missing_roots:
        raise FileNotFoundError("Search roots do not exist: {}".format(missing_roots))
    index = _waveform_index(roots)
    clean_rows = {}
    scenario_rows = {condition: {snr: {} for snr in SNRS}
                     for condition in ("noisy", "denoised")}

    csv_paths = sorted({path for root in roots for path in root.rglob("*.csv")})
    for path in csv_paths:
        layout = _csv_layout(path)
        if not layout or "emd" in path.as_posix().lower():
            continue
        columns, id_column, path_column = layout
        fold_column = _column(columns, FOLD_COLUMNS)
        snr_column = _column(columns, SNR_COLUMNS)
        condition_column = _column(columns, CONDITION_COLUMNS)
        path_condition = _condition(path.as_posix())
        if fold_column and "ptbxl_database" in path.name.lower():
            path_condition = "clean"
        path_snr = _snr_from_text(path.as_posix())
        with path.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                try:
                    record_id = _integer(row[id_column])
                except (TypeError, ValueError):
                    continue
                if fold_column and row.get(fold_column, "").strip():
                    try:
                        if _integer(row[fold_column]) != 10:
                            continue
                    except ValueError:
                        continue
                condition = _condition(row.get(condition_column, "")) if condition_column else path_condition
                if condition == "clean" and fold_column:
                    clean_rows.setdefault(record_id, (row[path_column], path))
                    continue
                snr = None
                if snr_column and row.get(snr_column, "").strip():
                    try:
                        snr = _integer(row[snr_column])
                    except ValueError:
                        snr = _snr_from_text(row[snr_column])
                snr = path_snr if snr is None else snr
                if condition in scenario_rows and snr in SNRS:
                    scenario_rows[condition][snr].setdefault(record_id, (row[path_column], path))

    if not clean_rows:
        raise ValueError("No fold-10 clean metadata with a WFDB path was discovered")
    fold10_ids = set(clean_rows)
    normalized = {"clean": []}
    for record_id, (record, source) in sorted(clean_rows.items()):
        resolved = _resolve_record(record, source, roots, index)
        normalized["clean"].append((record_id, "", str(resolved)))

    for condition in ("noisy", "denoised"):
        normalized[condition] = []
        for snr in SNRS:
            ids = set(scenario_rows[condition][snr])
            missing = fold10_ids - ids
            extra = ids - fold10_ids
            if missing:
                raise ValueError("{} SNR {} fold-10 ID coverage mismatch: {} missing; examples={}".format(
                    condition, snr, len(missing), sorted(missing)[:10]))
            if extra:
                print("Ignoring {} non-fold-10 {} records at SNR {}".format(
                    len(extra), condition, snr))
            for record_id in sorted(fold10_ids):
                record, source = scenario_rows[condition][snr][record_id]
                resolved = _resolve_record(record, source, roots, index)
                normalized[condition].append((record_id, snr, str(resolved)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    wfdb_roots = {}
    for condition, rows in normalized.items():
        manifest = output_dir / "{}_fold10_manifest.csv".format(condition)
        with manifest.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(("ecg_id", "snr_db", "record_path"))
            writer.writerows(rows)
        manifests[condition] = str(manifest.resolve())
        parents = [str(Path(row[2]).parent) for row in rows]
        wfdb_roots[condition] = os.path.commonpath(parents)

    config = {"fold": 10, "snrs_db": list(SNRS), "fold10_record_count": len(fold10_ids),
              "manifests": manifests, "wfdb_roots": wfdb_roots}
    config_path = output_dir / "original_models_benchmark_data.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print("Validated {} fold-10 IDs for clean, noisy, and denoised data at SNRs {}".format(
        len(fold10_ids), list(SNRS)))
    print("Wrote data configuration: {}".format(config_path))
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Safely prepare original-model benchmark data.")
    parser.add_argument("--archive", action="append", default=[])
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    roots = [Path(root) for root in args.search_root if Path(root).exists()]
    roots.extend(inspect_and_extract(archive, args.workspace) for archive in args.archive)
    discover_and_validate(roots, args.output_dir)


if __name__ == "__main__":
    main()
