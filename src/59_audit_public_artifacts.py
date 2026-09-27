"""Read-only audit of tracked personal paths and likely credentials.

The frozen historical lines are allowed only by exact file, line and line hash.
No path value or credential candidate is emitted by this program.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/public_artifact_privacy_audit_v1/baseline.json"

# Construct the personal-home prefix without embedding an example home path in
# the auditor itself. This keeps the audit scoped to real paths in tracked data.
PERSONAL_PATH = re.compile(
    rb"(?<![A-Za-z0-9])/(?:" + b"|".join((b"Users", b"home"))
    + rb")/[A-Za-z0-9._-]+"
)

CREDENTIAL_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "openai_key": re.compile(rb"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{32,}\b"),
    "bearer_token": re.compile(
        rb"\bBearer[ \t]+(?:eyJ[A-Za-z0-9_-]{20,}(?:\.[A-Za-z0-9_-]+){1,2}|[A-Za-z0-9_-]{32,})\b",
        re.IGNORECASE,
    ),
    "named_secret": re.compile(
        rb"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret)\b"
        rb"[ \t]*[:=][ \t]*[\"']?([A-Za-z0-9_./+-]{16,})",
        re.IGNORECASE,
    ),
    "password_assignment": re.compile(
        rb"\b(?:password|passwd|pwd)\b[ \t]*[:=][ \t]*[\"']?([^\s\"',;}{]{8,})",
        re.IGNORECASE,
    ),
}
PLACEHOLDER_PASSWORDS = {
    b"placeholder", b"changeme", b"yourpassword", b"password123",
    b"example123", b"testpassword", b"your_password",
}


def tracked_files(root: Path = ROOT) -> list[str]:
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    return sorted(p.decode("utf-8", errors="surrogateescape") for p in paths if p)


def scan_line(file: str, number: int, line: bytes) -> tuple[dict | None, list[dict]]:
    found = list(PERSONAL_PATH.finditer(line))
    path_entry = None
    if found:
        kinds = Counter("macos_home" if m.group().startswith(b"/Users/") else "linux_home" for m in found)
        path_entry = {
            "file": file,
            "line": number,
            "line_sha256": hashlib.sha256(line).hexdigest(),
            "path_types": dict(sorted(kinds.items())),
        }
    credentials = []
    for kind, pattern in CREDENTIAL_PATTERNS.items():
        for match in pattern.finditer(line):
            if kind == "password_assignment" and match.group(1).lower() in PLACEHOLDER_PASSWORDS:
                continue
            if kind == "named_secret" and any(
                CREDENTIAL_PATTERNS[specific].fullmatch(match.group(1))
                for specific in ("aws_access_key", "github_token", "openai_key")
            ):
                continue  # A recognisable key is one finding, not two.
            credentials.append({"file": file, "line": number, "type": kind})
    return path_entry, credentials


def scan_contents(contents: dict[str, bytes]) -> tuple[list[dict], list[dict]]:
    paths, credentials = [], []
    for file, data in sorted(contents.items()):
        for number, line in enumerate(data.splitlines(), 1):
            path, secrets = scan_line(file, number, line)
            if path is not None:
                paths.append(path)
            credentials.extend(secrets)
    return paths, credentials


def scan_repository(root: Path = ROOT) -> tuple[list[dict], list[dict], int]:
    files = tracked_files(root)
    contents = {name: (root / name).read_bytes() for name in files}
    paths, credentials = scan_contents(contents)
    return paths, credentials, len(files)


def assess(paths: list[dict], credentials: list[dict], baseline: dict) -> dict:
    known = {json.dumps(e, sort_keys=True) for e in baseline["entries"]}
    current = {json.dumps(e, sort_keys=True) for e in paths}
    additions = [json.loads(e) for e in sorted(current - known)]
    missing = [json.loads(e) for e in sorted(known - current)]
    return {
        "passed": not additions and not missing and not credentials,
        "known_historical_path_files": len({e["file"] for e in paths if json.dumps(e, sort_keys=True) in known}),
        "known_historical_path_hits": sum(sum(e["path_types"].values()) for e in paths if json.dumps(e, sort_keys=True) in known),
        "new_personal_path_hits": sum(sum(e["path_types"].values()) for e in additions),
        "changed_or_missing_baseline_lines": len(missing),
        "credential_hits": len(credentials),
        "credential_types": dict(sorted(Counter(e["type"] for e in credentials).items())),
        # Locations contain no secret values or personal path strings.
        "new_personal_path_locations": [{"file": e["file"], "line": e["line"]} for e in additions],
        "credential_locations": credentials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    assert baseline["schema_version"] == 1
    paths, credentials, files = scan_repository()
    result = assess(paths, credentials, baseline)
    result["tracked_files_scanned"] = files
    result["baseline"] = str(args.baseline.relative_to(ROOT))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
