"""List and retrieve explicitly selected tables from an authorized Delta Sharing profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import pyarrow.parquet as pq
import yaml

DEFAULT_OUTPUT_DIRECTORY = Path("data/volve")
DISK_RESERVE_BYTES = 512 * 1024 * 1024


class RetrievalError(Exception):
    """An expected, safe-to-report retrieval failure."""


@dataclass(frozen=True)
class AssetId:
    """A fully qualified Delta Sharing table identity."""

    share: str
    schema: str
    table: str

    @property
    def value(self) -> str:
        return f"{self.share}.{self.schema}.{self.table}"


@dataclass(frozen=True)
class Selection:
    """Validated user selection and local output settings."""

    assets: tuple[AssetId, ...]
    field_mappings: dict[str, str]
    output_directory: Path
    output_format: str
    overwrite_policy: str


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_sharing_client(profile_path: Path) -> Any:
    """Create the supported Delta Sharing client without exposing profile data."""
    try:
        import delta_sharing
    except ImportError as exc:
        raise RetrievalError("Delta Sharing dependencies are not installed.") from exc

    try:
        return delta_sharing.SharingClient(str(profile_path))
    except Exception as exc:  # Connector exceptions may contain credentials or signed URLs.
        raise RetrievalError(f"Could not open the share profile ({type(exc).__name__}).") from None


def validate_profile_path(value: str) -> Path:
    """Validate the location of the local profile without reading or printing it."""
    path = Path(value).expanduser()
    if not path.is_file():
        raise RetrievalError("The share profile path does not identify a readable file.")
    if not os.access(path, os.R_OK):
        raise RetrievalError("The share profile file is not readable by this process.")
    return path.resolve()


def asset_id_from_table(table: Any) -> AssetId:
    """Convert a connector table descriptor to a validated asset identifier."""
    values = (
        getattr(table, "share", None),
        getattr(table, "schema", None),
        getattr(table, "name", None),
    )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise RetrievalError("The connector returned a table without a complete share/schema/name.")
    return AssetId(*(value.strip() for value in values))


def _schema_summary(metadata: Any) -> list[dict[str, Any]] | None:
    """Extract only field names, types, and nullability from table metadata."""
    schema_string = getattr(metadata, "schema_string", None)
    if not isinstance(schema_string, str):
        return None
    try:
        schema = json.loads(schema_string)
    except json.JSONDecodeError:
        return None

    fields = schema.get("fields")
    if not isinstance(fields, list):
        return None
    result: list[dict[str, Any]] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        result.append(
            {
                "name": field.get("name"),
                "type": field.get("type"),
                "nullable": field.get("nullable"),
            }
        )
    return result


def _metadata_size(metadata: Any) -> int | None:
    """Return provider-reported snapshot size when available."""
    size = getattr(metadata, "size", None)
    return size if isinstance(size, int) and size >= 0 else None


def _metadata_file_count(metadata: Any) -> int | None:
    """Return provider-reported file count when available."""
    count = getattr(metadata, "num_files", getattr(metadata, "numFiles", None))
    return count if isinstance(count, int) and count >= 0 else None


def resolve_field_mappings(
    mappings: dict[str, str], inventory: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Resolve exact source field names without guessing ambiguous matches."""
    fields_by_asset = {
        entry["asset_id"]: {
            field.get("name")
            for field in (entry.get("schema_fields") or [])
            if isinstance(field, dict) and isinstance(field.get("name"), str)
        }
        for entry in inventory.get("assets", [])
    }
    resolved: dict[str, dict[str, Any]] = {}
    for role, source_field in mappings.items():
        matches = [
            asset_id for asset_id, fields in fields_by_asset.items() if source_field in fields
        ]
        if len(matches) == 1:
            resolved[role] = {
                "source_field": source_field,
                "asset_id": matches[0],
                "readiness": "present",
            }
        elif len(matches) > 1:
            resolved[role] = {
                "source_field": source_field,
                "asset_id": None,
                "readiness": "unresolved",
                "reason": "ambiguous field occurs in multiple selected/catalog assets",
            }
        else:
            resolved[role] = {
                "source_field": source_field,
                "asset_id": None,
                "readiness": "unresolved",
                "reason": "field was not observed in catalog metadata",
            }
    return resolved


def _metadata_for_table(client: Any, asset: AssetId) -> Any | None:
    """Fetch table metadata only; this does not read table rows."""
    try:
        return client.table(asset.value).metadata(use_delta_format=True)
    except Exception:
        try:
            return client.table(asset.value).metadata(use_delta_format=False)
        except Exception:
            return None


def catalog_inventory(client: Any) -> dict[str, Any]:
    """Enumerate tables and safe schema metadata without downloading table rows."""
    try:
        tables = list(client.list_all_tables())
    except Exception as exc:
        raise RetrievalError(
            f"Could not enumerate the authorized catalog ({type(exc).__name__})."
        ) from None

    shares: list[dict[str, Any]] = []
    if not tables:
        # Preserve visibility information when a share is accessible but has no
        # exposed schemas/tables; also use the documented APIs as a fallback.
        try:
            for share in client.list_shares():
                share_name = getattr(share, "name", None)
                schemas = client.list_schemas(share)
                schema_names = [getattr(schema, "name", None) for schema in schemas]
                shares.append({"name": share_name, "schemas": schema_names})
                for schema in schemas:
                    tables.extend(client.list_tables(schema))
        except Exception as exc:
            raise RetrievalError(
                f"Could not enumerate catalog schemas ({type(exc).__name__})."
            ) from None
    else:
        visible: dict[str, set[str]] = {}
        for table in tables:
            share_name = getattr(table, "share", None)
            schema_name = getattr(table, "schema", None)
            if isinstance(share_name, str) and isinstance(schema_name, str):
                visible.setdefault(share_name, set()).add(schema_name)
        shares = [
            {"name": name, "schemas": sorted(schema_names)}
            for name, schema_names in sorted(visible.items())
        ]

    assets: list[dict[str, Any]] = []
    for table in tables:
        asset = asset_id_from_table(table)
        metadata = _metadata_for_table(client, asset)
        format_info = getattr(metadata, "format", None) if metadata is not None else None
        provider = getattr(format_info, "provider", None)
        assets.append(
            {
                "asset_id": asset.value,
                "asset_kind": "unclassified",
                "share": asset.share,
                "schema": asset.schema,
                "table": asset.table,
                "format": provider or "unknown",
                "schema_fields": _schema_summary(metadata) if metadata is not None else None,
                "access_state": "visible",
                "estimated_size_bytes": _metadata_size(metadata) if metadata is not None else None,
                "estimated_file_count": (
                    _metadata_file_count(metadata) if metadata is not None else None
                ),
                "metadata_available": metadata is not None,
            }
        )

    return {"listed_at": utc_now(), "shares": shares, "assets": assets}


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON atomically with owner-only permissions where supported."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def run_list(profile_path: Path, inventory_out: Path | None) -> int:
    """List authorized table metadata; do not download data rows."""
    client = create_sharing_client(profile_path)
    inventory = catalog_inventory(client)
    if inventory_out is None:
        json.dump(inventory, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _write_json_atomic(inventory_out.expanduser(), inventory)
        print(f"Listed {len(inventory['assets'])} assets; metadata saved to {inventory_out}.")
    return 0


def load_selection(path: Path, output_override: Path | None = None) -> Selection:
    """Load a non-secret YAML selection and reject ambiguous or unsafe entries."""
    try:
        with path.expanduser().open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as exc:
        raise RetrievalError(f"Could not read the selection file ({type(exc).__name__}).") from None
    except yaml.YAMLError as exc:
        raise RetrievalError(f"The selection YAML is invalid ({type(exc).__name__}).") from None

    if not isinstance(data, dict):
        raise RetrievalError("The selection file must contain a YAML mapping.")
    raw_assets = data.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise RetrievalError("The selection must contain at least one explicit asset.")

    assets: list[AssetId] = []
    for index, value in enumerate(raw_assets, start=1):
        if not isinstance(value, dict):
            raise RetrievalError(f"Selection entry {index} must be a mapping.")
        components = (value.get("share"), value.get("schema"), value.get("table"))
        if any(not isinstance(part, str) or not part.strip() or "*" in part for part in components):
            raise RetrievalError(
                f"Selection entry {index} needs exact share, schema, and table names."
            )
        assets.append(AssetId(*(part.strip() for part in components)))

    if len({asset.value for asset in assets}) != len(assets):
        raise RetrievalError("The selection contains duplicate assets.")

    raw_mappings = data.get("field_mappings", {})
    if not isinstance(raw_mappings, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_mappings.items()
    ):
        raise RetrievalError(
            "field_mappings must map string workflow roles to string source fields."
        )

    output_format = data.get("output_format", "parquet")
    if output_format != "parquet":
        raise RetrievalError("Only Parquet output is supported by this retrieval utility.")
    overwrite_policy = data.get("overwrite_policy", "fail")
    if overwrite_policy != "fail":
        raise RetrievalError(
            "overwrite_policy must be 'fail'; existing files are never replaced silently."
        )

    configured_output = data.get("output_directory", str(DEFAULT_OUTPUT_DIRECTORY))
    if not isinstance(configured_output, str) or not configured_output.strip():
        raise RetrievalError("output_directory must be a non-empty local path.")
    output_directory = output_override or Path(configured_output).expanduser()
    return Selection(
        assets=tuple(assets),
        field_mappings=dict(raw_mappings),
        output_directory=output_directory,
        output_format=output_format,
        overwrite_policy=overwrite_policy,
    )


def _safe_component(value: str) -> str:
    """Percent-encode a provider identifier for use as one path component."""
    return quote(value, safe="") or "_"


def output_path_for(root: Path, asset: AssetId) -> Path:
    """Create a collision-safe output path and ensure it remains under root."""
    resolved_root = root.expanduser().resolve()
    relative = Path(
        _safe_component(asset.share),
        _safe_component(asset.schema),
        f"{_safe_component(asset.table)}.parquet",
    )
    candidate = (resolved_root / relative).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise RetrievalError("A selected asset generated an unsafe output path.")
    return candidate


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load an existing manifest or return a new non-secret manifest structure."""
    if not path.exists():
        return {"manifest_version": 1, "assets": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RetrievalError(
            "The existing manifest is unreadable or invalid; it was not overwritten."
        ) from None
    if (
        not isinstance(value, dict)
        or value.get("manifest_version") != 1
        or not isinstance(value.get("assets"), dict)
    ):
        raise RetrievalError(
            "The existing manifest has an unsupported structure; it was not overwritten."
        )
    return value


def _file_sha256(path: Path) -> str:
    """Compute a file checksum without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _available_size_guard(directory: Path, next_batch_bytes: int) -> None:
    """Stop a download before it consumes the reserved local disk space."""
    free_bytes = shutil.disk_usage(directory).free
    if free_bytes - next_batch_bytes < DISK_RESERVE_BYTES:
        raise RetrievalError(
            "Insufficient free disk space to continue while preserving a 512 MiB safety reserve."
        )


def _write_asset(client: Any, asset: AssetId, destination: Path) -> dict[str, Any]:
    """Stream one table to a temporary Parquet file and finalize only on success."""
    if destination.exists():
        raise RetrievalError(
            "The output file already exists; choose another output directory or review/remove it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    table_client = client.table(asset.value)
    try:
        metadata = table_client.metadata(use_delta_format=True)
    except Exception:
        try:
            metadata = table_client.metadata(use_delta_format=False)
        except Exception as exc:
            raise RetrievalError(f"Could not read table metadata ({type(exc).__name__}).") from None

    estimated_size = _metadata_size(metadata)
    if estimated_size is not None:
        free_bytes = shutil.disk_usage(destination.parent).free
        if estimated_size > max(0, free_bytes - DISK_RESERVE_BYTES):
            raise RetrievalError(
                f"Provider reports this table is about {estimated_size / (1024 ** 3):.2f} GiB; "
                "that exceeds available disk space while preserving the safety reserve."
            )

    temporary_path = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if temporary_path.exists():
        temporary_path.unlink()
    row_count = 0
    try:
        reader = table_client.to_record_batch_reader()
        with pq.ParquetWriter(temporary_path, reader.schema) as writer:
            for batch in reader:
                _available_size_guard(destination.parent, batch.nbytes)
                writer.write_batch(batch)
                row_count += batch.num_rows
        os.chmod(temporary_path, 0o600)
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            raise RetrievalError(
                "The output file appeared during retrieval; it was not overwritten."
            ) from None
        temporary_path.unlink()
    except RetrievalError:
        temporary_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # Do not expose connector exception text or URLs.
        temporary_path.unlink(missing_ok=True)
        raise RetrievalError(
            f"Transfer or Parquet serialization failed ({type(exc).__name__})."
        ) from None

    stat = destination.stat()
    fields = _schema_summary(metadata)
    return {
        "asset_id": asset.value,
        "source_schema": fields,
        "output_relative_path": destination.name,
        "output_format": "parquet",
        "retrieved_at": utc_now(),
        "file_size_bytes": stat.st_size,
        "checksum": _file_sha256(destination),
        "row_count": row_count,
        "estimated_source_size_bytes": estimated_size,
        "status": "complete",
    }


def run_download(profile_path: Path, selection_path: Path, output_override: Path | None) -> int:
    """Retrieve selected assets and atomically update a non-secret manifest."""
    selection = load_selection(selection_path, output_override)
    output_root = selection.output_directory.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = output_root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    client = create_sharing_client(profile_path)
    inventory = catalog_inventory(client)
    available_assets = {entry["asset_id"] for entry in inventory["assets"]}

    invalid = [asset.value for asset in selection.assets if asset.value not in available_assets]
    if invalid:
        raise RetrievalError(
            "Selected asset(s) are not currently visible to this profile: " + ", ".join(invalid)
        )

    selected_ids = {asset.value for asset in selection.assets}
    selected_inventory = {
        "assets": [entry for entry in inventory["assets"] if entry["asset_id"] in selected_ids]
    }
    mapping_resolution = resolve_field_mappings(selection.field_mappings, selected_inventory)
    manifest["field_mappings"] = mapping_resolution
    for role, result in mapping_resolution.items():
        if result["readiness"] != "present":
            print(
                f"Field mapping {role!r} remains unresolved: {result['reason']}.",
                file=sys.stderr,
            )

    failed = False
    for asset in selection.assets:
        destination = output_path_for(output_root, asset)
        print(f"Retrieving {asset.value} ...", file=sys.stderr)
        try:
            entry = _write_asset(client, asset, destination)
            entry["output_relative_path"] = destination.relative_to(output_root).as_posix()
            manifest["assets"][asset.value] = entry
            _write_json_atomic(manifest_path, manifest)
            print(
                f"Retrieved {asset.value}: {entry['file_size_bytes']} bytes, "
                f"{entry['row_count']} rows -> {entry['output_relative_path']}",
                file=sys.stderr,
            )
        except RetrievalError as exc:
            failed = True
            print(f"Failed {asset.value}: {exc}", file=sys.stderr)

    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the retrieval CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="List and retrieve explicitly selected tables from a Delta Sharing profile."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="List available assets without downloading rows."
    )
    list_parser.add_argument(
        "--profile", required=True, help="Path to the local .share profile file."
    )
    list_parser.add_argument(
        "--inventory-out", type=Path, help="Optional path for non-secret inventory JSON."
    )

    download_parser = subparsers.add_parser(
        "download", help="Retrieve explicitly selected assets as Parquet."
    )
    download_parser.add_argument(
        "--profile", required=True, help="Path to the local .share profile file."
    )
    download_parser.add_argument(
        "--selection", required=True, type=Path, help="Non-secret YAML asset selection file."
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Approved output directory (defaults to config or data/volve/).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the selected CLI operation and report safe, actionable failures."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profile_path = validate_profile_path(args.profile)
        if args.command == "list":
            return run_list(profile_path, args.inventory_out)
        return run_download(profile_path, args.selection, args.output_dir)
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: filesystem operation failed ({type(exc).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
