"""Synthetic path and credential fixtures; no experiment data is recomputed."""

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_audit", ROOT / "src/59_audit_public_artifacts.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PublicArtifactPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads(audit.BASELINE.read_text())

    def test_current_tracked_baseline_passes(self):
        paths, credentials, _ = audit.scan_repository()
        result = audit.assess(paths, credentials, self.baseline)
        self.assertTrue(result["passed"])
        self.assertEqual(result["known_historical_path_files"], 155)
        self.assertEqual(result["known_historical_path_hits"], 228)

    def test_new_macos_home_rejected(self):
        fixture = ("/" + "Users" + "/new_user/project/run.json").encode()
        paths, credentials = audit.scan_contents({"results/new.json": fixture})
        result = audit.assess(paths, credentials, {"entries": []})
        self.assertFalse(result["passed"])
        self.assertEqual(result["new_personal_path_hits"], 1)

    def test_new_linux_home_rejected(self):
        fixture = ("/" + "home" + "/new_user/project/run.json").encode()
        paths, credentials = audit.scan_contents({"results/new.json": fixture})
        self.assertEqual(audit.assess(paths, credentials, {"entries": []})["new_personal_path_hits"], 1)

    def test_simulated_api_key_rejected(self):
        fixture = ("api_key = sk-" + "A" * 40).encode()
        paths, credentials = audit.scan_contents({"results/new.json": fixture})
        result = audit.assess(paths, credentials, {"entries": []})
        self.assertFalse(result["passed"])
        self.assertEqual(result["credential_types"], {"openai_key": 1})

    def test_sequence_token_is_not_credential(self):
        fixture = b"F5 sequence token_count=200; sequence token UNKNOWN=1; token_stats matched"
        paths, credentials = audit.scan_contents({"results/new.json": fixture})
        self.assertTrue(audit.assess(paths, credentials, {"entries": []})["passed"])

    def test_generic_named_secret_rejected(self):
        fixture = ("access_token = " + "Z" * 36).encode()
        paths, credentials = audit.scan_contents({"results/new.json": fixture})
        self.assertEqual(audit.assess(paths, credentials, {"entries": []})["credential_types"], {"named_secret": 1})

    def test_changed_historical_line_is_not_whitelisted(self):
        old = self.baseline["entries"][0]
        changed = {**old, "line_sha256": "0" * 64}
        result = audit.assess([changed], [], {"entries": [old]})
        self.assertFalse(result["passed"])
        self.assertEqual(result["new_personal_path_hits"], sum(old["path_types"].values()))


if __name__ == "__main__":
    unittest.main()
