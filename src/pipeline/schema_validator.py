"""Schema validation utilities using jsonschema with $ref resolution."""
import json
import os
from pathlib import Path
from jsonschema import Draft202012Validator, RefResolver

SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"

def _build_resolver():
    store = {}
    base_uri = f"file://{SCHEMA_DIR}/"
    local_https_base = "https://neuro-symbolic-nodule-followup.local/schemas/"
    for schema_file in SCHEMA_DIR.glob("*.json"):
        with open(schema_file) as f:
            schema = json.load(f)
        if "$id" in schema:
            store[schema["$id"]] = schema
        store[f"{base_uri}{schema_file.name}"] = schema
        store[f"{local_https_base}{schema_file.name}"] = schema
        store[schema_file.name] = schema
    return RefResolver(base_uri, {}, store=store)

def load_schema(name: str) -> dict:
    path = SCHEMA_DIR / name
    if not path.suffix:
        path = path.with_suffix(".json")
    with open(path) as f:
        return json.load(f)

def validate_instance(instance: dict, schema_name: str) -> list[str]:
    schema = load_schema(schema_name)
    resolver = _build_resolver()
    validator = Draft202012Validator(schema, resolver=resolver)
    errors = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")
    return errors

def is_valid(instance: dict, schema_name: str) -> bool:
    return len(validate_instance(instance, schema_name)) == 0
