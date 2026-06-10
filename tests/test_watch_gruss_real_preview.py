from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_gruss_real_preview.py"
SPEC = importlib.util.spec_from_file_location("watch_gruss_real_preview", SCRIPT_PATH)
watch_gruss_real_preview = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watch_gruss_real_preview)


VALID_ENV = {
    "DOGBOT_DATA_PROVIDER": "gruss_excel",
    "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
    "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "false",
    "DOGBOT_GRUSS_REAL_PREVIEW": "true",
}


class WatchGrussRealPreviewTests(unittest.TestCase):
    def test_refuses_if_real_orders_are_enabled(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_ENABLE_REAL_ORDERS="true")

        with self.assertRaisesRegex(RuntimeError, "ENABLE_REAL_ORDERS=true est interdit"):
            watch_gruss_real_preview.validate_real_preview_environment(env)

    def test_requires_preview_true(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_PREVIEW="false")

        with self.assertRaisesRegex(RuntimeError, "REAL_PREVIEW=true est obligatoire"):
            watch_gruss_real_preview.validate_real_preview_environment(env)

    def test_waits_only_outside_active_pre_post_milestones(self) -> None:
        for seconds in (20, 15, 10, 5, 0):
            with self.subTest(seconds=seconds):
                self.assertIsNone(watch_gruss_real_preview.countdown_wait_reason(seconds, seconds))
        self.assertEqual(
            watch_gruss_real_preview.countdown_wait_reason(3, 3),
            "wait: countdown_seconds=3 next_milestone=0 execution_phase=POST",
        )

    def test_valid_preview_environment_is_accepted(self) -> None:
        watch_gruss_real_preview.validate_real_preview_environment(VALID_ENV)


if __name__ == "__main__":
    unittest.main()
