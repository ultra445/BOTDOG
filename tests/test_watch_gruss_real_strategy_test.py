from __future__ import annotations

import csv
import io
import importlib.util
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
from dogbot.excel_strategy_loader import (
    CONDITIONS_COLUMNS,
    GLOBAL_SETTINGS_COLUMNS,
    STAKE_PROFILES_COLUMNS,
    STRATEGIES_COLUMNS,
    VARIABLES_COLUMNS,
    _write_xlsx,
)
from dogbot.gruss.gruss_orders import make_order_intent
from dogbot.gruss.gruss_real_orders import (
    GrussExcelOrderProvider,
    GrussRealOrderContext,
    is_terminal_bet_status,
    is_valid_bet_ref,
    normalise_gruss_bet_ref,
    strip_gruss_ref_suffix,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_gruss_real_strategy_test.py"
SPEC = importlib.util.spec_from_file_location("watch_gruss_real_strategy_test", SCRIPT_PATH)
watch_gruss_real_strategy_test = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watch_gruss_real_strategy_test)


VALID_ENV = {
    "DOGBOT_DATA_PROVIDER": "gruss_excel",
    "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
    "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
    "DOGBOT_GRUSS_REAL_TEST_MODE": "true",
    "DOGBOT_GRUSS_REAL_MAX_ORDERS": "1",
    "DOGBOT_GRUSS_REAL_MAX_STAKE": "2",
    "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE": "2",
    "DOGBOT_GRUSS_REAL_PREVIEW": "false",
    "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
    "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
    "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "0",
    "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS": "0",
    "DOGBOT_POST_COMMAND_CELLS_CLEAR_DELAY_MS": "0",
    "DOGBOT_POST_BET_REF_REQUIRED": "false",
    "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "false",
    "DOGBOT_STRATEGIES_EXCEL_ENABLED": "true",
}
PRE_LADDER_REAL_ENV = {
    **VALID_ENV,
    "DOGBOT_PRE_LADDER_ENABLED": "true",
    "DOGBOT_PRE_LADDER_PREVIEW": "false",
    "DOGBOT_PRE_LADDER_STEPS": "45,32,20,14",
    "DOGBOT_PRE_LADDER_REAL_MAX_LADDERS": "1",
    "DOGBOT_PRE_LADDER_REAL_REQUIRE_BET_REF_FOR_REPLACE": "true",
    "DOGBOT_PRE_LADDER_REAL_STOP_IF_NO_BET_REF": "true",
    "DOGBOT_PRE_LADDER_REAL_NO_STACKING": "true",
    "DOGBOT_GRUSS_PRE_BATCH_WRITE_SLEEP_MS": "0",
}


class FakeBridge:
    def __init__(self) -> None:
        self.write_calls = []
        self.clear_calls = []
        self.cells = {}

    def connect_open_workbook(self):
        return True

    def is_workbook_visible(self):
        return True

    def has_sheet(self, sheet_name):
        return sheet_name in {"WIN", "PLACE"}

    def sheet_names(self):
        return ["WIN", "PLACE"]

    def read_cell(self, sheet_name, address):
        key = (sheet_name, address)
        if key in self.cells:
            return self.cells[key]
        if address == "N3":
            return {"WIN": "win-1", "PLACE": "place-1"}[sheet_name]
        if address == "D2":
            return "00:00:20"
        if address == "F2":
            return "Active"
        return None

    def read_range(self, sheet_name, address):
        return [[f"{index}. Runner {index}"] for index in range(1, 11)]

    def write_cells(self, sheet_name, cells, *, allow_write=False):
        plan = tuple(cells)
        self.write_calls.append((sheet_name, plan, allow_write))
        for address, value in plan:
            self.cells[(sheet_name, address)] = value
        return [address for address, _ in plan]

    def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, command_columns=None, allow_clear=False):
        prepared = tuple(addresses)
        self.clear_calls.append((sheet_name, prepared, trigger_column, allow_clear))
        for address in prepared:
            self.cells[(sheet_name, address)] = None
        return list(prepared)


class FakeBetRefBridge(FakeBridge):
    def __init__(self, *, write_bet_ref_after_initial: bool = True) -> None:
        super().__init__()
        self.write_bet_ref_after_initial = write_bet_ref_after_initial

    def write_cells(self, sheet_name, cells, *, allow_write=False):
        written = super().write_cells(sheet_name, cells, allow_write=allow_write)
        plan = dict(cells)
        trigger_items = [
            (address, value)
            for address, value in plan.items()
            if str(address).upper().startswith("Q")
        ]
        for address, value in trigger_items:
            row = str(address).upper().replace("Q", "", 1)
            if value in {"BACK", "LAY"}:
                self.cells[(sheet_name, f"W{row}")] = 0.0
                if self.write_bet_ref_after_initial:
                    self.cells[(sheet_name, f"T{row}")] = f"4320000000{int(row):02d}"
            elif value in {"BACKR", "LAYR"}:
                self.cells[(sheet_name, f"T{row}")] = f"4320000000{int(row):02d}N"
                self.cells[(sheet_name, f"W{row}")] = 0.0
        return written


class FakeTemporaryComRejectBridge(FakeBetRefBridge):
    def __init__(self, *, failures_before_success: int | None) -> None:
        super().__init__()
        self.failures_before_success = failures_before_success

    def write_cells(self, sheet_name, cells, *, allow_write=False):
        if self.failures_before_success is None or self.failures_before_success > 0:
            if self.failures_before_success is not None:
                self.failures_before_success -= 1
            raise RuntimeError(-2147418111, "L'appel a ete rejete par l'appele.")
        return super().write_cells(sheet_name, cells, allow_write=allow_write)


class FakeTemporaryRunnerRangeRejectBridge(FakeBetRefBridge):
    def __init__(
        self,
        *,
        failures_before_success: int,
        message: str = "This object does not support enumeration",
    ) -> None:
        super().__init__()
        self.failures_before_success = failures_before_success
        self.message = message

    def read_range(self, sheet_name, address):
        if sheet_name == "PLACE" and address == "A5:A84" and self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError(-2147418111, self.message)
        return super().read_range(sheet_name, address)


class FakeTemporaryCleanupRejectBridge(FakeBetRefBridge):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_failures = 1

    def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, command_columns=None, allow_clear=False):
        if self.cleanup_failures > 0:
            self.cleanup_failures -= 1
            raise RuntimeError(-2147418111, "L'appel a ete rejete par l'appele.")
        return super().clear_trigger_cells(
            sheet_name,
            addresses,
            trigger_column=trigger_column,
            command_columns=command_columns,
            allow_clear=allow_clear,
        )


class FakeDelayedBetRefBridge(FakeBetRefBridge):
    def __init__(self, *, reads_before_ref: int = 2) -> None:
        super().__init__(write_bet_ref_after_initial=False)
        self.reads_before_ref = reads_before_ref
        self.pending_bet_refs = {}
        self.bet_ref_reads = {}
        self.events = []

    def write_cells(self, sheet_name, cells, *, allow_write=False):
        written = FakeBridge.write_cells(self, sheet_name, cells, allow_write=allow_write)
        plan = dict(cells)
        for address, value in plan.items():
            if not str(address).upper().startswith("Q"):
                continue
            row = str(address).upper().replace("Q", "", 1)
            self.events.append(("write", sheet_name, f"Q{row}", value))
            if value in {"BACK", "LAY"}:
                self.pending_bet_refs[(sheet_name, f"T{row}")] = f"4320000000{int(row):02d}"
                self.cells[(sheet_name, f"W{row}")] = 0.0
            elif value in {"BACKR", "LAYR"}:
                self.cells[(sheet_name, f"T{row}")] = f"4320000000{int(row):02d}N"
                self.cells[(sheet_name, f"W{row}")] = 0.0
        return written

    def read_cell(self, sheet_name, address):
        key = (sheet_name, str(address).upper())
        if key in self.pending_bet_refs:
            self.events.append(("read", sheet_name, str(address).upper(), None))
            reads = self.bet_ref_reads.get(key, 0) + 1
            self.bet_ref_reads[key] = reads
            if reads <= self.reads_before_ref:
                return None
            value = self.pending_bet_refs.pop(key)
            self.cells[key] = value
            return value
        return super().read_cell(sheet_name, address)


class FakePendingReplaceBetRefBridge(FakeBetRefBridge):
    def __init__(self, *, reads_before_replace_ref: int | None = 1) -> None:
        super().__init__(write_bet_ref_after_initial=True)
        self.reads_before_replace_ref = reads_before_replace_ref
        self.replace_ref_reads = {}

    def write_cells(self, sheet_name, cells, *, allow_write=False):
        written = FakeBridge.write_cells(self, sheet_name, cells, allow_write=allow_write)
        plan = dict(cells)
        for address, value in plan.items():
            if not str(address).upper().startswith("Q"):
                continue
            row = str(address).upper().replace("Q", "", 1)
            if value in {"BACK", "LAY"}:
                self.cells[(sheet_name, f"T{row}")] = f"4320000000{int(row):02d}"
                self.cells[(sheet_name, f"W{row}")] = 0.0
            elif value in {"BACKR", "LAYR"}:
                self.cells[(sheet_name, f"T{row}")] = "PENDINGR"
                self.cells[(sheet_name, f"W{row}")] = 0.0
                self.replace_ref_reads[(sheet_name, f"T{row}")] = 0
        return written

    def read_cell(self, sheet_name, address):
        key = (sheet_name, str(address).upper())
        if key in self.replace_ref_reads:
            reads = self.replace_ref_reads[key] + 1
            self.replace_ref_reads[key] = reads
            if self.reads_before_replace_ref is None or reads <= self.reads_before_replace_ref:
                return "PENDINGR"
            row = str(address).upper().replace("T", "", 1)
            value = f"4329000000{int(row):02d}N"
            self.cells[key] = value
            del self.replace_ref_reads[key]
            return value
        return super().read_cell(sheet_name, address)


class FakeMatchedMetricsBridge(FakeBetRefBridge):
    def write_cells(self, sheet_name, cells, *, allow_write=False):
        written = super().write_cells(sheet_name, cells, allow_write=allow_write)
        for address, value in dict(cells).items():
            if str(address).upper().startswith("Q") and value in {"BACK", "LAY", "BACKR", "LAYR", "CANCEL"}:
                row = str(address).upper().replace("Q", "", 1)
                self.cells[(sheet_name, f"V{row}")] = 2.72
                self.cells[(sheet_name, f"W{row}")] = 1.5
                self.cells[(sheet_name, f"X{row}")] = -3.0
        return written


class FakeMatchedNoBetRefBridge(FakeBetRefBridge):
    def __init__(self) -> None:
        super().__init__(write_bet_ref_after_initial=False)

    def write_cells(self, sheet_name, cells, *, allow_write=False):
        written = super().write_cells(sheet_name, cells, allow_write=allow_write)
        for address, value in dict(cells).items():
            if str(address).upper().startswith("Q") and value in {"BACK", "LAY", "BACKR", "LAYR"}:
                row = str(address).upper().replace("Q", "", 1)
                self.cells[(sheet_name, f"V{row}")] = 2.72
                self.cells[(sheet_name, f"W{row}")] = 1.5
                self.cells[(sheet_name, f"X{row}")] = -3.0
        return written


class FakeSelectionsBetRefBridge(FakeBetRefBridge):
    def __init__(self, rows_by_attempt: list[list[list[object]]]) -> None:
        super().__init__(write_bet_ref_after_initial=False)
        self.rows_by_attempt = list(rows_by_attempt)
        self.read_sheet_calls = []

    def has_sheet(self, sheet_name):
        return sheet_name in {"WIN", "PLACE", "WIN_Selections", "PLACE_Selections"}

    def sheet_names(self):
        return ["WIN", "PLACE", "WIN_Selections", "PLACE_Selections"]

    def read_sheet(self, sheet_name, rows=80, columns=50):
        self.read_sheet_calls.append((sheet_name, rows, columns))
        if sheet_name != "PLACE_Selections":
            return []
        if not self.rows_by_attempt:
            return []
        return self.rows_by_attempt.pop(0)


class FakeProcessedStore:
    def __init__(self) -> None:
        self.seen = set()
        self.mark_calls = []

    def has_seen(self, key):
        return key in self.seen

    def mark_seen(self, key, win_market_id, place_market_id):
        self.mark_calls.append((key, win_market_id, place_market_id))
        self.seen.add(key)


def _intent(index: int, *, stake: float = 5.0):
    return make_order_intent(
        provider="gruss_excel",
        market_type="PLACE",
        market_id="place-1",
        parent_id="1",
        runner_name=f"Runner {index}",
        trap=index,
        side="BACK",
        order_type="LIMIT",
        price=3.0,
        stake=stake,
        strategy_id=f"BACK_PLACE_{100 + index}",
        course_id="parent:1",
        dry_run=True,
    )


def _pre_ladder_intent(
    step: str,
    *,
    ladder_id: str = "ladder-1",
    trap: int = 1,
    side: str = "BACK",
    stake: float = 5.0,
    strategy_id: str | None = None,
):
    resolved_strategy_id = strategy_id or ("BACK_PLACE_101" if side == "BACK" else "LAY_PLACE_301")
    return make_order_intent(
        provider="gruss_excel",
        market_type="PLACE",
        market_id="place-1",
        parent_id="1",
        runner_name=f"Runner {trap}",
        trap=trap,
        side=side,
        order_type="LIMIT",
        price=3.0,
        stake=stake,
        strategy_id=resolved_strategy_id,
        course_id="parent:1",
        dry_run=True,
        selection_id=str(trap),
        execution_phase="PRE",
        triggered_systems=resolved_strategy_id,
        triggered_prices="3.0",
        pre_ladder=True,
        ladder_id=ladder_id,
        ladder_step=step,
        ladder_tracking_key=f"{ladder_id}:tracking",
        matched_stake=0.0,
    )


def _context():
    return GrussRealOrderContext(
        validation_ok=True,
        tradable=True,
        region="UK",
        countdown_seconds=2,
        course="parent:1",
        win_market_id="win-1",
        place_market_id="place-1",
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


LEGACY_MAX_LADDERS_REASON = "pre_ladder_real_" + "max_ladders_must_be_1"


class WatchGrussRealStrategyTestTests(unittest.TestCase):
    def test_gruss_bet_ref_helpers_accept_only_real_numeric_refs(self) -> None:
        for value in ("432049736284", "432049736284.0", "432049736284N", "432049736284.0N", 432049736284, 432049736284.0):
            with self.subTest(value=value):
                self.assertTrue(is_valid_bet_ref(value))

        for value in ("", "PENDING", "PENDINGR", "CANCELLED", "LAPSED", "RESULT_LOST", "RESULT_WON", "VARIOUS", "BR-5", "SEL-LAY-1"):
            with self.subTest(value=value):
                self.assertFalse(is_valid_bet_ref(value))

        self.assertEqual(strip_gruss_ref_suffix("432049736284N"), "432049736284")
        self.assertEqual(strip_gruss_ref_suffix("432049736284.0N"), "432049736284")
        self.assertEqual(strip_gruss_ref_suffix("432049736284"), "432049736284")
        self.assertEqual(normalise_gruss_bet_ref("432049736284.0"), "432049736284")
        self.assertEqual(normalise_gruss_bet_ref("432049736284.0N"), "432049736284N")
        self.assertTrue(is_terminal_bet_status("RESULT_LOST"))
        self.assertTrue(is_terminal_bet_status("LAPSED"))
        self.assertFalse(is_terminal_bet_status("432049736284"))

    def test_legacy_max_ladders_must_be_one_reason_is_absent_from_runtime_code(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runtime_files = [*list((root / "src").rglob("*.py")), *list((root / "scripts").rglob("*.py"))]

        offenders = [
            str(path.relative_to(root))
            for path in runtime_files
            if LEGACY_MAX_LADDERS_REASON in path.read_text(encoding="utf-8")
        ]

        self.assertEqual(offenders, [])

    def test_valid_environment_is_accepted(self) -> None:
        self.assertEqual(
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(VALID_ENV),
            (1, 2.0, 2.0),
        )

    def test_real_environment_requires_excel_strategy_workbook_enabled(self) -> None:
        env = dict(VALID_ENV, DOGBOT_STRATEGIES_EXCEL_ENABLED="false")

        with self.assertRaisesRegex(RuntimeError, "DOGBOT_STRATEGIES_EXCEL_ENABLED=true est obligatoire"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_strategy_workbook_startup_summary_prints_functional_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            report = Path(tmp) / "load_report.csv"
            _write_xlsx(
                path,
                {
                    "Strategies": [
                        STRATEGIES_COLUMNS,
                        [
                            "TRUE",
                            "BACK_PLACE_901",
                            "Back place functional",
                            "PLACE",
                            "BACK",
                            "PRE",
                            "LIMIT",
                            "LIMIT_THEO_FUNC",
                            "",
                            "",
                            "",
                            "place_theorique",
                            "DYNAMIC",
                            "AGGRESSIVE",
                            "PLACE_BSP_THEN_LTP",
                            "FALSE",
                            "",
                            "TRUE",
                            "EDGE",
                            "",
                            "FALSE",
                            "TRUE",
                            "EXCEL",
                            "",
                            "EXCEL",
                            "TEST",
                            "BACK_STANDARD",
                            "10",
                            "",
                            "BACK_PLACE_UK_T1_SMOOTH",
                        ],
                    ],
                    "Conditions": [
                        CONDITIONS_COLUMNS,
                        ["TRUE", "BACK_PLACE_901", "1", "market_type", "=", "PLACE", ""],
                    ],
                    "Variables": [
                        VARIABLES_COLUMNS,
                        ["market_type", "Market type", "text", "LIVE", "ctx", "PLACE", "TRUE"],
                        ["place_theorique", "Place theo", "number", "LIVE", "ctx", "2.0", "TRUE"],
                    ],
                    "StakeProfiles": [
                        STAKE_PROFILES_COLUMNS,
                        ["BACK_STANDARD", "VARIABLE", "1", "5", "2", "1", "Back"],
                    ],
                    "GlobalSettings": [
                        GLOBAL_SETTINGS_COLUMNS,
                        ["PRE_LADDER_STEPS", "52,38,26,16", "Default PRE ladder"],
                    ],
                    "README": [["README"]],
                },
            )
            env = dict(
                VALID_ENV,
                DOGBOT_STRATEGIES_EXCEL_PATH=str(path),
                DOGBOT_STRATEGIES_EXCEL_REPORT_PATH=str(report),
            )
            output = io.StringIO()

            with redirect_stdout(output):
                watch_gruss_real_strategy_test.print_strategy_workbook_startup_summary(env)

        text = output.getvalue()
        self.assertIn("strategy_workbook_startup enabled=true", text)
        self.assertIn("active_strategies=1", text)
        self.assertIn("first_strategy_ids=BACK_PLACE_901", text)
        self.assertIn("LIMIT_THEO_FUNC_count=1", text)
        self.assertIn("function_name_count=1", text)
        self.assertIn("functional_strategy_ids=BACK_PLACE_901", text)

    def test_variable_stakes_environment_allows_five_euro_cap_without_force_stake(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="5",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
        )

        self.assertEqual(
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env),
            (1, 5.0, None),
        )

    def test_variable_stakes_environment_refuses_cap_above_five_without_override(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="7",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
        )

        with self.assertRaisesRegex(RuntimeError, "DOGBOT_GRUSS_REAL_MAX_STAKE > 5 en mode variable"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_variable_stakes_environment_allows_seven_with_explicit_override(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="7",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
            DOGBOT_GRUSS_REAL_ALLOW_VARIABLE_STAKE_OVER_5="true",
            DOGBOT_GRUSS_REAL_VARIABLE_STAKE_HARD_CAP="10",
        )

        self.assertEqual(
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env),
            (1, 7.0, None),
        )

    def test_variable_stakes_environment_refuses_above_hard_cap_even_with_override(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="20",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
            DOGBOT_GRUSS_REAL_ALLOW_VARIABLE_STAKE_OVER_5="true",
            DOGBOT_GRUSS_REAL_VARIABLE_STAKE_HARD_CAP="10",
        )

        with self.assertRaisesRegex(RuntimeError, "DOGBOT_GRUSS_REAL_MAX_STAKE exceeds DOGBOT_GRUSS_REAL_VARIABLE_STAKE_HARD_CAP"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_force_stake_mode_still_rejects_seven_euro_cap(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="false",
            DOGBOT_GRUSS_REAL_MAX_STAKE="7",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="7",
            DOGBOT_GRUSS_REAL_ALLOW_VARIABLE_STAKE_OVER_5="true",
        )

        with self.assertRaisesRegex(RuntimeError, "DOGBOT_GRUSS_REAL_MAX_STAKE doit etre exactement 2 en mode force_stake"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_real_strategy_promotes_eligible_pre_ladder_preview_rows_when_ladder_is_armed(self) -> None:
        rows = [
            {
                "status": "PRE_LADDER_PREVIEW",
                "execution_phase": "PRE",
                "market_type": "PLACE",
                "ladder_id": "ladder-1",
                "ladder_step": "1/4",
                "current_ladder_price": "6.8",
                "current_step_stake": "2.0",
            }
        ]

        prepared, promoted = watch_gruss_real_strategy_test.prepare_trade_rows_for_real_provider(
            rows,
            PRE_LADDER_REAL_ENV,
        )

        self.assertEqual(promoted, 1)
        self.assertEqual(prepared[0]["status"], "PRE_LADDER_REAL_READY")
        self.assertEqual(prepared[0]["real_strategy_original_status"], "PRE_LADDER_PREVIEW")
        self.assertEqual(prepared[0]["real_strategy_promotion_reason"], "pre_ladder_real_enabled")
        self.assertEqual(rows[0]["status"], "PRE_LADDER_PREVIEW")

    def test_real_strategy_does_not_promote_pre_ladder_preview_rows_unless_ladder_is_armed(self) -> None:
        row = {
            "status": "PRE_LADDER_PREVIEW",
            "execution_phase": "PRE",
            "market_type": "PLACE",
            "ladder_id": "ladder-1",
            "ladder_step": "1/4",
            "current_ladder_price": "6.8",
            "current_step_stake": "2.0",
        }

        for env in (
            VALID_ENV,
            dict(PRE_LADDER_REAL_ENV, DOGBOT_PRE_LADDER_PREVIEW="true"),
        ):
            with self.subTest(env=env):
                prepared, promoted = watch_gruss_real_strategy_test.prepare_trade_rows_for_real_provider(
                    [row],
                    env,
                )

                self.assertEqual(promoted, 0)
                self.assertEqual(prepared[0]["status"], "PRE_LADDER_PREVIEW")

    def test_real_strategy_promotion_requires_complete_place_pre_ladder_fields(self) -> None:
        base = {
            "status": "PRE_LADDER_PREVIEW",
            "execution_phase": "PRE",
            "market_type": "PLACE",
            "ladder_id": "ladder-1",
            "ladder_step": "1/4",
            "current_ladder_price": "6.8",
            "current_step_stake": "2.0",
        }

        for missing in ("ladder_id", "ladder_step", "current_ladder_price", "current_step_stake"):
            with self.subTest(missing=missing):
                row = dict(base, **{missing: ""})
                prepared, promoted = watch_gruss_real_strategy_test.prepare_trade_rows_for_real_provider(
                    [row],
                    PRE_LADDER_REAL_ENV,
                )

                self.assertEqual(promoted, 0)
                self.assertEqual(prepared[0]["status"], "PRE_LADDER_PREVIEW")

    def test_real_strategy_configures_pre_ladder_even_when_env_default_is_disabled(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_PRE_LADDER_ENABLED="false",
            DOGBOT_PRE_LADDER_PREVIEW="true",
        )

        applied = watch_gruss_real_strategy_test.configure_real_pre_ladder_for_strategy_test(env)

        self.assertEqual(applied["DOGBOT_PRE_LADDER_ENABLED"], "true")
        self.assertEqual(applied["DOGBOT_PRE_LADDER_PREVIEW"], "false")
        self.assertEqual(applied["DOGBOT_PRE_LADDER_STEPS"], "52,38,26,16")
        self.assertEqual(applied["DOGBOT_PRE_POST_INDEPENDENT"], "true")
        self.assertEqual(applied["DOGBOT_PRE_CANCEL_BEFORE_POST"], "false")
        self.assertEqual(applied["DOGBOT_PRE_CANCEL_ONLY_IF_POST_PENDING"], "false")
        self.assertEqual(applied["DOGBOT_POST_SKIP_IF_PRE_MATCHED"], "false")
        self.assertEqual(applied["DOGBOT_PRE_LADDER_REAL_MAX_LADDERS"], "50")
        self.assertEqual(
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env),
            (1, 2.0, 2.0),
        )

    def test_real_strategy_preserves_configured_pre_ladder_steps(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_PRE_LADDER_ENABLED="false",
            DOGBOT_PRE_LADDER_PREVIEW="true",
            DOGBOT_PRE_LADDER_STEPS="40,30,20,14",
        )

        applied = watch_gruss_real_strategy_test.configure_real_pre_ladder_for_strategy_test(env)

        self.assertEqual(applied["DOGBOT_PRE_LADDER_STEPS"], "40,30,20,14")
        self.assertEqual(env["DOGBOT_PRE_LADDER_STEPS"], "40,30,20,14")

    def test_real_strategy_applies_default_pre_ladder_steps_when_absent(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_PRE_LADDER_ENABLED="false",
            DOGBOT_PRE_LADDER_PREVIEW="true",
        )
        env.pop("DOGBOT_PRE_LADDER_STEPS", None)

        applied = watch_gruss_real_strategy_test.configure_real_pre_ladder_for_strategy_test(env)

        self.assertEqual(applied["DOGBOT_PRE_LADDER_STEPS"], "52,38,26,16")
        self.assertEqual(env["DOGBOT_PRE_LADDER_STEPS"], "52,38,26,16")

    def test_real_strategy_preserves_configured_pre_ladder_max_ladders(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_PRE_LADDER_ENABLED="false",
            DOGBOT_PRE_LADDER_PREVIEW="true",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="50",
        )

        applied = watch_gruss_real_strategy_test.configure_real_pre_ladder_for_strategy_test(env)

        self.assertEqual(applied["DOGBOT_PRE_LADDER_REAL_MAX_LADDERS"], "50")
        self.assertEqual(env["DOGBOT_PRE_LADDER_REAL_MAX_LADDERS"], "50")
        self.assertEqual(
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env),
            (1, 2.0, 2.0),
        )

    def test_real_strategy_loads_env_file_before_pre_ladder_defaults(self) -> None:
        env: dict[str, str] = {}
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "DOGBOT_PRE_LADDER_REAL_MAX_LADDERS=50",
                        "DOGBOT_GRUSS_REAL_VARIABLE_STAKES=true",
                        "DOGBOT_GRUSS_REAL_MAX_STAKE=5",
                        "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE=",
                    ]
                ),
                encoding="utf-8",
            )

            loaded = watch_gruss_real_strategy_test.load_strategy_test_env_file(env, env_path)
            applied = watch_gruss_real_strategy_test.configure_real_pre_ladder_for_strategy_test(env)

        self.assertEqual(loaded["DOGBOT_PRE_LADDER_REAL_MAX_LADDERS"], "50")
        self.assertEqual(env["DOGBOT_PRE_LADDER_REAL_MAX_LADDERS"], "50")
        self.assertEqual(applied["DOGBOT_PRE_LADDER_REAL_MAX_LADDERS"], "50")
        self.assertEqual(env["DOGBOT_GRUSS_REAL_VARIABLE_STAKES"], "true")
        self.assertEqual(env["DOGBOT_GRUSS_REAL_MAX_STAKE"], "5")
        self.assertEqual(env["DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE"], "")

    def test_real_strategy_env_file_does_not_override_exported_max_ladders(self) -> None:
        env = {"DOGBOT_PRE_LADDER_REAL_MAX_LADDERS": "7"}
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("DOGBOT_PRE_LADDER_REAL_MAX_LADDERS=50\n", encoding="utf-8")

            loaded = watch_gruss_real_strategy_test.load_strategy_test_env_file(env, env_path)
            applied = watch_gruss_real_strategy_test.configure_real_pre_ladder_for_strategy_test(env)

        self.assertNotIn("DOGBOT_PRE_LADDER_REAL_MAX_LADDERS", loaded)
        self.assertEqual(env["DOGBOT_PRE_LADDER_REAL_MAX_LADDERS"], "7")
        self.assertEqual(applied["DOGBOT_PRE_LADDER_REAL_MAX_LADDERS"], "7")

    def test_real_strategy_promotes_no_better_range_pre_ladder_rows_after_configuration(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_PRE_LADDER_ENABLED="false",
            DOGBOT_PRE_LADDER_PREVIEW="true",
        )
        watch_gruss_real_strategy_test.configure_real_pre_ladder_for_strategy_test(env)
        rows = [
            {
                "status": "PRE_LADDER_PREVIEW",
                "reason": "no_better_back_ladder_range",
                "execution_phase": "PRE",
                "market_type": "PLACE",
                "ladder_id": "ladder-1",
                "ladder_step": "1/4",
                "current_ladder_price": "5.0",
                "current_step_stake": "2.0",
                "no_better_ladder_range_reason": "no_better_back_ladder_range",
            }
        ]

        prepared, promoted = watch_gruss_real_strategy_test.prepare_trade_rows_for_real_provider(
            rows,
            env,
        )

        self.assertEqual(promoted, 1)
        self.assertEqual(prepared[0]["status"], "PRE_LADDER_REAL_READY")
        self.assertEqual(prepared[0]["reason"], "no_better_back_ladder_range")
        self.assertEqual(prepared[0]["current_ladder_price"], "5.0")

    def test_real_strategy_updates_trades_status_after_pre_ladder_write(self) -> None:
        intent = _pre_ladder_intent("1/4")
        context = replace(_context(), countdown_seconds=45, milestone_seen=45)
        processed_key = watch_gruss_real_strategy_test._real_provider_processed_key(intent, context)
        fieldnames = [
            "status",
            "reason",
            "course_id",
            "parent_id",
            "market_id",
            "selection_id",
            "trap",
            "side",
            "market_type",
            "execution_phase",
            "pre_ladder",
            "ladder_step",
            "ladder_id",
            "ladder_tracking_key",
        ]
        row = {
            "status": "PRE_LADDER_REAL_READY",
            "reason": "pre_ladder_real_ready",
            "course_id": "parent:1",
            "parent_id": "1",
            "market_id": "place-1",
            "selection_id": "1",
            "trap": "1",
            "side": "BACK",
            "market_type": "PLACE",
            "execution_phase": "PRE",
            "pre_ladder": "True",
            "ladder_step": "1/4",
            "ladder_id": "ladder-1",
            "ladder_tracking_key": "ladder-1:tracking",
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trades_20260614.csv"
            _write_csv(path, fieldnames, [row])
            changed = watch_gruss_real_strategy_test.update_trade_rows_with_real_results(
                path,
                0,
                [intent],
                [
                    SimpleNamespace(
                        processed_key=processed_key,
                        status="GRUSS_PRE_LADDER_WRITTEN",
                        reason="pre_ladder_step_written",
                        direct_lim_order_written=False,
                    )
                ],
                context,
            )
            updated = _read_rows(path)

        self.assertEqual(changed, 1)
        self.assertEqual(updated[0]["status"], "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(updated[0]["reason"], "pre_ladder_step_written")

    def test_real_strategy_updates_trades_status_for_direct_lim_write(self) -> None:
        intent = _pre_ladder_intent("1/4")
        context = replace(_context(), countdown_seconds=45, milestone_seen=45)
        processed_key = watch_gruss_real_strategy_test._real_provider_processed_key(intent, context)
        fieldnames = [
            "status",
            "reason",
            "course_id",
            "market_id",
            "selection_id",
            "side",
            "market_type",
            "execution_phase",
            "pre_ladder",
            "ladder_step",
            "ladder_id",
        ]
        row = {
            "status": "PRE_LADDER_REAL_READY",
            "reason": "no_better_back_ladder_range",
            "course_id": "parent:1",
            "market_id": "place-1",
            "selection_id": "1",
            "side": "BACK",
            "market_type": "PLACE",
            "execution_phase": "PRE",
            "pre_ladder": "True",
            "ladder_step": "1/4",
            "ladder_id": "ladder-1",
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trades_20260614.csv"
            _write_csv(path, fieldnames, [row])
            changed = watch_gruss_real_strategy_test.update_trade_rows_with_real_results(
                path,
                0,
                [intent],
                [
                    SimpleNamespace(
                        processed_key=processed_key,
                        status="GRUSS_PRE_LADDER_WRITTEN",
                        reason="pre_ladder_step_written",
                        direct_lim_order_written=True,
                    )
                ],
                context,
            )
            updated = _read_rows(path)

        self.assertEqual(changed, 1)
        self.assertEqual(updated[0]["status"], "DIRECT_LIM_ORDER_WRITTEN")
        self.assertEqual(updated[0]["reason"], "pre_ladder_step_written")

    def test_direct_lim_ready_row_writes_initial_order_before_marking_written(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        store = FakeProcessedStore()
        intent = replace(
            _pre_ladder_intent("1/4"),
            price=5.0,
            stake=2.0,
            direct_lim_order_planned=True,
            direct_lim_order_written=False,
            ladder_disabled_lim_not_in_ladder_direction=True,
            no_replace_steps_for_direct_lim=True,
        )

        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_PRE_POST_INDEPENDENT="false",
            DOGBOT_PRE_CANCEL_BEFORE_POST="true",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[intent],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(bridge.write_calls[0][1], (("R5", 5.0), ("S5", 2.0), ("Q5", "BACK")))
        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertTrue(results[0].direct_lim_order_written)
        self.assertEqual(rows[0]["direct_lim_order_planned"], "True")
        self.assertEqual(rows[0]["direct_lim_order_written"], "True")
        self.assertEqual(rows[0]["status"], "GRUSS_PRE_LADDER_WRITTEN")

    def test_direct_lim_batch_processes_every_selection_or_logs_rejection(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        store = FakeProcessedStore()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="10",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        intents = [
            replace(
                _pre_ladder_intent("1/4", ladder_id=f"ladder-{trap}", trap=trap),
                price=1.5 + (trap / 10.0),
                stake=2.0,
                direct_lim_order_planned=True,
                direct_lim_order_written=False,
                ladder_disabled_lim_not_in_ladder_direction=True,
                no_replace_steps_for_direct_lim=True,
            )
            for trap in range(1, 7)
        ]

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual(len(results), 6)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(bridge.write_calls), 6)
        self.assertEqual([result.status for result in results], ["GRUSS_PRE_LADDER_WRITTEN"] * 6)
        self.assertEqual([row["selection_id"] for row in rows], [str(trap) for trap in range(1, 7)])
        self.assertEqual([row["excel_row"] for row in rows], [str(row) for row in range(5, 11)])
        self.assertTrue(all(row["direct_lim_order_planned"] == "True" for row in rows))
        self.assertTrue(all(row["direct_lim_order_written"] == "True" for row in rows))
        self.assertTrue(all(row["direct_lim_candidates_count"] == "6" for row in rows))
        self.assertEqual([row["direct_lim_candidate_index"] for row in rows], [str(index) for index in range(1, 7)])
        self.assertTrue(all(row["direct_lim_provider_called"] == "True" for row in rows))
        self.assertTrue(all(row["direct_lim_batch_processed_count"] == "6" for row in rows))
        self.assertTrue(all(row["direct_lim_written_count"] == "6" for row in rows))
        self.assertTrue(all(row["direct_lim_rejected_count"] == "0" for row in rows))

    def test_first_pre_batch_candidate_is_written_not_rejected_by_countdown_guard(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:47"
        store = FakeProcessedStore()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_PRE_LADDER_STEPS="47,30,21,14",
            DOGBOT_GRUSS_REAL_MAX_ORDERS="10",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        first = replace(
            _pre_ladder_intent(
                "1/4",
                ladder_id="ladder-trap-1",
                trap=1,
                strategy_id="BACK_PLACE_214",
                stake=2.0,
            ),
            price=3.0,
            direct_lim_order_planned=True,
            direct_lim_order_written=False,
            ladder_disabled_lim_not_in_ladder_direction=True,
            no_replace_steps_for_direct_lim=True,
        )
        rest = [
            replace(
                _pre_ladder_intent("1/4", ladder_id=f"ladder-trap-{trap}", trap=trap, stake=2.0),
                price=1.5 + (trap / 10.0),
                direct_lim_order_planned=True,
                direct_lim_order_written=False,
                ladder_disabled_lim_not_in_ladder_direction=True,
                no_replace_steps_for_direct_lim=True,
            )
            for trap in range(2, 7)
        ]

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[first, *rest],
                context=replace(_context(), countdown_seconds=47, milestone_seen=47),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual(len(results), 6)
        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[0].reason, "pre_ladder_step_written")
        self.assertNotIn("countdown_above_3_seconds", [result.reason for result in results])
        self.assertEqual(rows[0]["selection_id"], "1")
        self.assertEqual(rows[0]["strategy_id"], "BACK_PLACE_214")
        self.assertEqual(rows[0]["order_index_in_batch"], "1")
        self.assertEqual(rows[0]["pre_batch_candidate_index"], "1")
        self.assertEqual(rows[0]["pre_batch_candidates_count"], "6")
        self.assertEqual(rows[0]["direct_lim_candidate_index"], "1")
        self.assertEqual(rows[0]["direct_lim_provider_called"], "True")
        self.assertEqual(len(bridge.write_calls), 6)

    def test_pre_ladder_excel_write_retry_recovers_temporary_com_rejection(self) -> None:
        bridge = FakeTemporaryComRejectBridge(failures_before_success=1)
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_EXCEL_WRITE_RETRIES="3",
            DOGBOT_GRUSS_EXCEL_WRITE_RETRY_BACKOFF_MS="0,0,0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(_pre_ladder_intent("1/4", stake=2.0), stake_forced=True, stake_original=2.0),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(result.reason, "pre_ladder_step_written")
        self.assertEqual(result.excel_write_attempt, 2)
        self.assertEqual(result.excel_write_retry_count, 1)
        self.assertEqual(result.excel_write_retry_backoff_ms, "0")
        self.assertEqual(result.excel_write_final_status, "written")
        self.assertTrue(result.excel_unavailable_recovered)
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(rows[0]["excel_write_attempt"], "2")
        self.assertEqual(rows[0]["excel_write_retry_count"], "1")
        self.assertEqual(rows[0]["excel_write_retry_backoff_ms"], "0")
        self.assertEqual(rows[0]["excel_write_final_status"], "written")
        self.assertEqual(rows[0]["excel_unavailable_recovered"], "True")

    def test_pre_ladder_mapping_retry_recovers_temporary_runner_range_error(self) -> None:
        bridge = FakeTemporaryRunnerRangeRejectBridge(failures_before_success=2)
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_EXCEL_COM_RETRIES="3",
            DOGBOT_GRUSS_EXCEL_COM_RETRY_BACKOFF_MS="0,0,0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(_pre_ladder_intent("1/4", stake=2.0), stake_forced=True, stake_original=2.0),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(result.mapping_attempt_count, 3)
        self.assertTrue(result.excel_unavailable_recovered)
        self.assertGreaterEqual(result.excel_com_retry_count, 0)
        self.assertEqual(rows[0]["mapping_attempt_count"], "3")
        self.assertEqual(rows[0]["excel_unavailable_recovered"], "True")
        self.assertNotEqual(rows[0]["excel_write_attempt"], "0")

    def test_pre_ladder_mapping_retry_recovers_mojibake_french_com_rejection(self) -> None:
        bridge = FakeTemporaryRunnerRangeRejectBridge(
            failures_before_success=1,
            message="Lâ€™appel a Ã©tÃ© rejetÃ© par lâ€™appelÃ©.",
        )
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_EXCEL_COM_RETRIES="3",
            DOGBOT_GRUSS_EXCEL_COM_RETRY_BACKOFF_MS="0,0,0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(_pre_ladder_intent("1/4", stake=2.0), stake_forced=True, stake_original=2.0),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(result.mapping_attempt_count, 2)
        self.assertTrue(result.excel_unavailable_recovered)
        self.assertEqual(rows[0]["mapping_attempt_count"], "2")
        self.assertEqual(rows[0]["excel_unavailable_recovered"], "True")

    def test_stale_command_cells_cleanup_retries_temporary_com_rejection(self) -> None:
        bridge = FakeTemporaryCleanupRejectBridge()
        bridge.cells[("PLACE", "Q5")] = "CANCEL"
        bridge.cells[("PLACE", "R5")] = 3.0
        bridge.cells[("PLACE", "S5")] = 2.0
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_EXCEL_COM_RETRIES="3",
            DOGBOT_GRUSS_EXCEL_COM_RETRY_BACKOFF_MS="0,0,0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            cleanup = provider.cleanup_stale_command_cells(reason="startup")
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(cleanup.get("failed"))
        self.assertIn("PLACE!Q5", cleanup["addresses"])
        self.assertIsNone(bridge.cells[("PLACE", "Q5")])
        self.assertIsNone(bridge.cells[("PLACE", "R5")])
        self.assertIsNone(bridge.cells[("PLACE", "S5")])
        self.assertEqual(bridge.cleanup_failures, 0)
        self.assertGreaterEqual(len(bridge.clear_calls), 1)
        self.assertEqual(rows[0]["startup_command_cells_cleanup_done"], "True")

    def test_pre_ladder_excel_write_retry_exhaustion_is_not_marked_processed(self) -> None:
        bridge = FakeTemporaryComRejectBridge(failures_before_success=None)
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        store = FakeProcessedStore()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_EXCEL_WRITE_RETRIES="2",
            DOGBOT_GRUSS_EXCEL_WRITE_RETRY_BACKOFF_MS="0,0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", stake=2.0)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual(results[0].status, "REJECTED_REAL")
        self.assertIn("excel_unavailable_after_retries", results[0].reason)
        self.assertEqual(results[0].excel_write_attempt, 3)
        self.assertEqual(results[0].excel_write_retry_count, 2)
        self.assertEqual(results[0].excel_write_retry_backoff_ms, "0;0")
        self.assertEqual(results[0].excel_write_final_status, "excel_unavailable_after_retries")
        self.assertFalse(results[0].excel_unavailable_recovered)
        self.assertEqual(store.mark_calls, [])
        self.assertEqual(rows[0]["status"], "REJECTED_REAL")
        self.assertEqual(rows[0]["excel_write_attempt"], "3")
        self.assertEqual(rows[0]["excel_write_retry_count"], "2")
        self.assertEqual(rows[0]["excel_write_retry_backoff_ms"], "0;0")
        self.assertEqual(rows[0]["excel_write_final_status"], "excel_unavailable_after_retries")

    def test_pre_batch_write_sleep_runs_between_pre_initial_candidates_only(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        store = FakeProcessedStore()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="10",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
            DOGBOT_GRUSS_PRE_BATCH_WRITE_SLEEP_MS="250",
        )
        intents = [
            _pre_ladder_intent("1/4", ladder_id=f"ladder-{trap}", trap=trap, stake=2.0)
            for trap in range(1, 4)
        ]

        with (
            TemporaryDirectory() as tmp,
            patch.dict("os.environ", env, clear=True),
            patch.object(watch_gruss_real_strategy_test.time, "sleep") as sleep_mock,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        self.assertFalse(already_processed)
        self.assertEqual([result.status for result in results], ["GRUSS_PRE_LADDER_WRITTEN"] * 3)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [0.25, 0.25])

    def test_pre_initial_batch_grace_allows_late_direct_lim_runners(self) -> None:
        class SlowBatchCountdownBridge(FakeBetRefBridge):
            def __init__(self) -> None:
                super().__init__()
                self.countdowns = [46, 43, 41, 39, 37, 35]
                self.write_index = 0

            def read_cell(self, sheet_name, address):
                if address == "D2":
                    index = min(self.write_index, len(self.countdowns) - 1)
                    return f"00:00:{self.countdowns[index]:02d}"
                return super().read_cell(sheet_name, address)

            def write_cells(self, sheet_name, cells, *, allow_write=False):
                written = super().write_cells(sheet_name, cells, allow_write=allow_write)
                if any(str(address).upper().startswith("Q") for address, _ in cells):
                    self.write_index += 1
                return written

        bridge = SlowBatchCountdownBridge()
        store = FakeProcessedStore()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="10",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
            DOGBOT_PRE_INITIAL_BATCH_WRITE_GRACE_SECONDS="10",
        )
        intents = [
            replace(
                _pre_ladder_intent("1/4", ladder_id=f"ladder-{trap}", trap=trap),
                price=1.5 + (trap / 10.0),
                stake=2.0,
                direct_lim_order_planned=True,
                direct_lim_order_written=False,
                ladder_disabled_lim_not_in_ladder_direction=True,
                no_replace_steps_for_direct_lim=True,
            )
            for trap in range(1, 7)
        ]

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual([result.status for result in results], ["GRUSS_PRE_LADDER_WRITTEN"] * 6)
        self.assertEqual(len(bridge.write_calls), 6)
        self.assertEqual([row["pre_batch_milestone_authorized"] for row in rows], ["True"] * 6)
        self.assertEqual([row["pre_batch_milestone_seconds"] for row in rows], ["45"] * 6)
        self.assertEqual([row["pre_batch_started_countdown_seconds"] for row in rows], ["45"] * 6)
        self.assertEqual([row["pre_batch_write_grace_seconds"] for row in rows], ["10"] * 6)
        self.assertEqual([row["pre_batch_candidate_index"] for row in rows], [str(index) for index in range(1, 7)])
        self.assertEqual([row["pre_batch_candidates_count"] for row in rows], ["6"] * 6)
        self.assertEqual([row["pre_batch_late_write_allowed"] for row in rows], ["True", "False", "False", "True", "True", "True"])
        self.assertEqual([row["pre_batch_late_write_seconds_after_start"] for row in rows], ["0", "2", "4", "6", "8", "10"])
        self.assertNotIn("pre_ladder_milestone_window_missed", [result.reason for result in results])

    def test_real_strategy_updates_trades_status_after_real_rejection(self) -> None:
        intent = _pre_ladder_intent("2/4")
        context = replace(_context(), countdown_seconds=32, milestone_seen=32)
        processed_key = watch_gruss_real_strategy_test._real_provider_processed_key(intent, context)
        fieldnames = [
            "status",
            "reason",
            "course_id",
            "market_id",
            "selection_id",
            "side",
            "market_type",
            "execution_phase",
            "pre_ladder",
            "ladder_step",
            "ladder_id",
        ]
        row = {
            "status": "PRE_LADDER_REAL_READY",
            "reason": "pre_ladder_real_ready",
            "course_id": "parent:1",
            "market_id": "place-1",
            "selection_id": "1",
            "side": "BACK",
            "market_type": "PLACE",
            "execution_phase": "PRE",
            "pre_ladder": "True",
            "ladder_step": "2/4",
            "ladder_id": "ladder-1",
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trades_20260614.csv"
            _write_csv(path, fieldnames, [row])
            changed = watch_gruss_real_strategy_test.update_trade_rows_with_real_results(
                path,
                0,
                [intent],
                [
                    SimpleNamespace(
                        processed_key=processed_key,
                        status="REJECTED_REAL",
                        reason="pre_ladder_initial_order_not_written_no_replace",
                        direct_lim_order_written=False,
                    )
                ],
                context,
            )
            updated = _read_rows(path)

        self.assertEqual(changed, 1)
        self.assertEqual(updated[0]["status"], "REJECTED_REAL")
        self.assertEqual(updated[0]["reason"], "pre_ladder_initial_order_not_written_no_replace")

    def test_real_strategy_does_not_promote_direct_lim_replace_steps(self) -> None:
        rows = [
            {
                "status": "PRE_LADDER_PREVIEW",
                "reason": "no_better_back_ladder_range",
                "execution_phase": "PRE",
                "market_type": "PLACE",
                "ladder_id": "ladder-1",
                "ladder_step": "2/4",
                "current_ladder_price": "5.0",
                "current_step_stake": "2.0",
                "ladder_prices_frozen": "5.0",
                "no_replace_steps_for_direct_lim": "True",
            }
        ]

        prepared, promoted = watch_gruss_real_strategy_test.prepare_trade_rows_for_real_provider(
            rows,
            PRE_LADDER_REAL_ENV,
        )

        self.assertEqual(promoted, 0)
        self.assertEqual(prepared[0]["status"], "PRE_LADDER_PREVIEW")

    def test_real_strategy_does_not_promote_constant_multi_step_ladder_plan(self) -> None:
        rows = [
            {
                "status": "PRE_LADDER_PREVIEW",
                "execution_phase": "PRE",
                "market_type": "PLACE",
                "ladder_id": "ladder-1",
                "ladder_step": "1/4",
                "current_ladder_price": "28.0",
                "current_step_stake": "2.0",
                "ladder_prices_frozen": "28.0|28.0|28.0|28.0",
            }
        ]

        prepared, promoted = watch_gruss_real_strategy_test.prepare_trade_rows_for_real_provider(
            rows,
            PRE_LADDER_REAL_ENV,
        )

        self.assertEqual(promoted, 0)
        self.assertEqual(prepared[0]["status"], "PRE_LADDER_PREVIEW")

    def test_real_strategy_processes_post_real_ready_rows_with_real_provider(self) -> None:
        bridge = FakeBridge()
        store = FakeProcessedStore()
        post_intent = replace(
            _intent(1, stake=2.0),
            execution_phase="POST",
            selection_id="1",
            dry_run=True,
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[post_intent],
                context=_context(),
                processed_store=store,
                key="race-1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(rows[0]["execution_phase"], "POST")
        self.assertEqual(rows[0]["status"], "GRUSS_REAL_WRITTEN")
        self.assertEqual(rows[0]["provider"], ORDER_PROVIDER_GRUSS_EXCEL_REAL)

    def test_post_at_t_minus_one_logs_cancel_attempt_then_writes_post(self) -> None:
        bridge = FakeBridge()
        store = FakeProcessedStore()
        post_intent = replace(
            _intent(1, stake=2.0),
            execution_phase="POST",
            selection_id="1",
            dry_run=True,
        )

        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_PRE_POST_INDEPENDENT="false",
            DOGBOT_PRE_CANCEL_BEFORE_POST="true",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[post_intent],
                context=replace(_context(), countdown_seconds=1, milestone_seen=1),
                processed_store=store,
                key="race-1|milestone=1|phase=POST",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual([result.reason for result in results], [
            "pre_cancel_skipped_no_active_pre_ladders",
            "excel_trigger_written",
        ])
        self.assertEqual([row["status"] for row in rows], [
            "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
            "GRUSS_REAL_WRITTEN",
        ])
        self.assertEqual(rows[0]["pre_cancel_attempted"], "True")
        self.assertEqual(rows[0]["pre_cancel_written"], "False")
        self.assertEqual(rows[0]["pre_cancel_skip_reason"], "pre_cancel_skipped_no_active_pre_ladders")
        self.assertEqual(rows[0]["countdown_seconds_at_cancel"], "1")
        self.assertEqual(rows[1]["execution_phase"], "POST")
        self.assertEqual(rows[1]["post_write_attempted"], "True")
        self.assertEqual(rows[1]["post_write_status"], "GRUSS_REAL_WRITTEN")
        self.assertEqual(len(bridge.write_calls), 1)

    def test_pre_post_independent_skips_cancel_and_writes_post_directly(self) -> None:
        bridge = FakeBridge()
        store = FakeProcessedStore()
        post_intent = replace(
            _intent(1, stake=2.0),
            execution_phase="POST",
            selection_id="1",
            dry_run=True,
        )
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_PRE_POST_INDEPENDENT="true",
            DOGBOT_PRE_CANCEL_BEFORE_POST="false",
            DOGBOT_PRE_CANCEL_ONLY_IF_POST_PENDING="false",
            DOGBOT_POST_SKIP_IF_PRE_MATCHED="false",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[post_intent],
                context=replace(_context(), countdown_seconds=1, milestone_seen=1),
                processed_store=store,
                key="race-1|milestone=1|phase=POST",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertFalse(results[0].pre_cancel_attempted)
        self.assertEqual(results[0].pre_cancel_skip_reason, "pre_post_independent")
        self.assertEqual(rows[0]["pre_post_independent"], "True")
        self.assertEqual(rows[0]["pre_cancel_required_before_post"], "False")
        self.assertEqual(rows[0]["pre_cancel_attempted"], "False")
        self.assertEqual(rows[0]["pre_cancel_skip_reason"], "pre_post_independent")
        self.assertEqual(rows[0]["post_provider_called"], "True")
        self.assertEqual(rows[0]["post_write_attempted"], "True")
        self.assertEqual([call[1][-1][1] for call in bridge.write_calls], ["BACK"])

    def test_post_batch_writes_all_candidates_before_long_polling(self) -> None:
        bridge = FakeBridge()
        store = FakeProcessedStore()
        sleep_write_counts: list[int] = []
        intents = [
            replace(_intent(index, stake=2.0), execution_phase="POST", selection_id=str(index), dry_run=True)
            for index in range(1, 6)
        ]
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_PRE_POST_INDEPENDENT="true",
            DOGBOT_PRE_CANCEL_BEFORE_POST="false",
            DOGBOT_PRE_CANCEL_ONLY_IF_POST_PENDING="false",
            DOGBOT_POST_SKIP_IF_PRE_MATCHED="false",
            DOGBOT_POST_BET_REF_REQUIRED="true",
            DOGBOT_POST_BET_REF_WAIT_MS="35",
            DOGBOT_POST_BET_REF_POLL_MS="10",
            DOGBOT_POST_SEND_SECONDS_BEFORE_OFF="12",
            DOGBOT_GRUSS_REAL_MAX_ORDERS="10",
        )

        def observe_sleep(_seconds):
            sleep_write_counts.append(len(bridge.write_calls))

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True), patch(
            "dogbot.gruss.gruss_real_orders.time.sleep",
            side_effect=observe_sleep,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, already_processed = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=12, milestone_seen=12),
                processed_store=store,
                key="race-1|milestone=12|phase=POST",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(already_processed)
        self.assertEqual(len(bridge.write_calls), 5)
        self.assertTrue(sleep_write_counts)
        self.assertEqual(sleep_write_counts[0], 5)
        self.assertEqual([row["post_batch_candidate_count"] for row in rows], ["5"] * 5)
        self.assertEqual([row["post_batch_written_count"] for row in rows], ["5"] * 5)
        self.assertTrue(all(row["post_batch_write_duration_ms"] != "" for row in rows))
        self.assertEqual([row["post_batch_confirmation_started"] for row in rows], ["True"] * 5)
        self.assertEqual([row["post_batch_runner_index"] for row in rows], ["1", "2", "3", "4", "5"])
        self.assertEqual([row["post_order_confirmed"] for row in rows], ["False"] * 5)
        self.assertTrue(all(result.status == "POST_WRITE_UNCONFIRMED" for result in results))

    def test_pre_cancel_before_post_writes_cancel_for_unmatched_active_ladder(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="10")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4"),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            provider.collect_pre_ladder_bet_refs([first])
            cancel_results = provider.cancel_pre_ladders_before_post(
                replace(_context(), countdown_seconds=1, milestone_seen=1)
            )

        self.assertEqual(first.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(len(cancel_results), 1)
        self.assertEqual(cancel_results[0].status, "GRUSS_PRE_CANCEL_BEFORE_POST_WRITTEN")
        self.assertEqual(cancel_results[0].reason, "pre_cancel_before_post_written")
        self.assertEqual(bridge.write_calls[-1][1], (("Q5", "CANCEL"),))
        self.assertEqual(cancel_results[0].trigger_mapping_name, "CANCEL")
        self.assertTrue(cancel_results[0].pre_cancel_attempted)
        self.assertTrue(cancel_results[0].pre_cancel_written)
        self.assertEqual(cancel_results[0].bet_ref_at_cancel, "432000000005")
        self.assertEqual(cancel_results[0].matched_stake_at_cancel, 0.0)
        self.assertEqual(cancel_results[0].countdown_seconds_at_cancel, 1)
        self.assertEqual(bridge.clear_calls[-1], ("PLACE", ("Q5", "R5", "S5"), "Q", True))
        self.assertIsNone(bridge.cells[("PLACE", "Q5")])
        self.assertIsNone(bridge.cells[("PLACE", "R5")])
        self.assertIsNone(bridge.cells[("PLACE", "S5")])
        self.assertEqual(bridge.cells[("PLACE", "T5")], "432000000005")
        self.assertEqual(cancel_results[0].command_cells_clear_reason, "command_cells_cleared")
        self.assertEqual(cancel_results[0].command_cells_clear_addresses, "Q5;R5;S5")

    def test_pre_cancel_before_post_skips_matched_ladder(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="10")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4"),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            provider.collect_pre_ladder_bet_refs([first])
            bridge.cells[("PLACE", "W5")] = 0.5
            cancel_results = provider.cancel_pre_ladders_before_post(
                replace(_context(), countdown_seconds=1, milestone_seen=1)
            )

        self.assertEqual(len(cancel_results), 1)
        self.assertEqual(cancel_results[0].status, "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED")
        self.assertEqual(cancel_results[0].reason, "pre_cancel_skipped_matched_stake_gt_zero")
        self.assertTrue(cancel_results[0].pre_cancel_attempted)
        self.assertFalse(cancel_results[0].pre_cancel_written)
        self.assertEqual(cancel_results[0].pre_cancel_skip_reason, "pre_cancel_skipped_matched_stake_gt_zero")
        self.assertEqual(cancel_results[0].bet_ref_at_cancel, "432000000005")
        self.assertEqual(cancel_results[0].matched_stake_at_cancel, 0.5)
        self.assertEqual(cancel_results[0].countdown_seconds_at_cancel, 1)
        self.assertEqual(bridge.write_calls[-1][1], (("R5", 3.0), ("S5", 2.0), ("Q5", "BACK")))

    def test_post_skip_if_pre_matched_rejects_post_after_cancel_attempt(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="10",
            DOGBOT_POST_SKIP_IF_PRE_MATCHED="true",
            DOGBOT_PRE_POST_INDEPENDENT="false",
            DOGBOT_PRE_CANCEL_BEFORE_POST="true",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4"),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            provider.collect_pre_ladder_bet_refs([first])
            bridge.cells[("PLACE", "W5")] = 0.5
            bridge.cells[("PLACE", "D2")] = "00:00:01"
            results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_intent(1, stake=2.0)],
                context=replace(_context(), countdown_seconds=1, milestone_seen=1),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=1|phase=POST",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        self.assertFalse(skipped)
        self.assertEqual([result.reason for result in results], ["pre_cancel_skipped_matched_stake_gt_zero", "post_skipped_pre_matched"])
        self.assertEqual(bridge.write_calls, [("PLACE", (("R5", 3.0), ("S5", 2.0), ("Q5", "BACK")), True)])

    def test_max_orders_is_required(self) -> None:
        env = dict(VALID_ENV)
        env.pop("DOGBOT_GRUSS_REAL_MAX_ORDERS")

        with self.assertRaisesRegex(RuntimeError, "DOGBOT_GRUSS_REAL_MAX_ORDERS est obligatoire"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_max_orders_must_be_a_positive_integer(self) -> None:
        for value in ("abc", "0", "-1"):
            with self.subTest(value=value):
                env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS=value)
                expected = "doit etre un entier" if value == "abc" else "doit etre un entier positif"

                with self.assertRaisesRegex(RuntimeError, expected):
                    watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_max_orders_one_and_five_hundred_are_accepted(self) -> None:
        for max_orders in (1, 500):
            with self.subTest(max_orders=max_orders):
                env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS=str(max_orders))

                self.assertEqual(
                    watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env),
                    (max_orders, 2.0, 2.0),
                )

    def test_strict_max_orders_one_message_is_absent_from_strategy_watcher(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("DOGBOT_GRUSS_REAL_MAX_ORDERS doit etre exactement 1", source)

    def test_force_test_modes_are_forbidden(self) -> None:
        for forbidden in (
            "DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT",
            "DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE",
        ):
            with self.subTest(forbidden=forbidden):
                env = dict(VALID_ENV, **{forbidden: "true"})
                with self.assertRaisesRegex(RuntimeError, f"{forbidden}=true est interdit"):
                    watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_preview_and_write_no_trigger_are_forbidden(self) -> None:
        for forbidden in ("DOGBOT_GRUSS_REAL_PREVIEW", "DOGBOT_GRUSS_WRITE_NO_TRIGGER"):
            with self.subTest(forbidden=forbidden):
                env = dict(VALID_ENV, **{forbidden: "true"})
                with self.assertRaisesRegex(RuntimeError, f"{forbidden}=true est interdit"):
                    watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_pre_ladder_real_requires_preview_false_when_enabled(self) -> None:
        env = dict(VALID_ENV, DOGBOT_PRE_LADDER_ENABLED="true", DOGBOT_PRE_LADDER_PREVIEW="true")

        with self.assertRaisesRegex(RuntimeError, "PRE_LADDER_PREVIEW=false"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_pre_ladder_real_forbids_hold_trigger_visual_test(self) -> None:
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST="true",
        )

        with self.assertRaisesRegex(RuntimeError, "HOLD_TRIGGER_FOR_VISUAL_TEST=true est interdit"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_pre_ladder_real_requires_positive_max_ladders(self) -> None:
        absent = dict(PRE_LADDER_REAL_ENV)
        absent.pop("DOGBOT_PRE_LADDER_REAL_MAX_LADDERS")

        with self.assertRaisesRegex(RuntimeError, "DOGBOT_PRE_LADDER_REAL_MAX_LADDERS est obligatoire"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(absent)

        for value in ("abc", "0", "-1"):
            with self.subTest(value=value):
                env = dict(PRE_LADDER_REAL_ENV, DOGBOT_PRE_LADDER_REAL_MAX_LADDERS=value)
                expected = "doit etre un entier" if value == "abc" else "doit etre un entier positif"

                with self.assertRaisesRegex(RuntimeError, expected):
                    watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_force_stake_two_requires_real_test_mode(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_TEST_MODE="false")

        with self.assertRaisesRegex(RuntimeError, "REAL_TEST_MODE=true est obligatoire"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_force_stake_must_be_exactly_two(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1")

        with self.assertRaisesRegex(RuntimeError, "REAL_TEST_FORCE_STAKE doit etre exactement 2 en mode force_stake"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_real_strategies_multiple_signals_write_one_and_reject_rest(self) -> None:
        bridge = FakeBridge()
        store = FakeProcessedStore()
        intents = [_intent(index, stake=5.0) for index in range(1, 5)]

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=_context(),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(skipped)
        self.assertEqual([result.status for result in results].count("GRUSS_REAL_WRITTEN"), 1)
        self.assertEqual([result.reason for result in results[1:]], ["max_orders_reached"] * 3)
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.0), ("S5", 2.0), ("Q5", "BACK")))
        self.assertEqual(store.mark_calls, [("parent:1|course:1|win:win-1|place:place-1|POST", "win-1", "place-1")])
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["status"], "GRUSS_REAL_WRITTEN")
        self.assertEqual([row["reason"] for row in rows[1:]], ["max_orders_reached"] * 3)

    def test_processed_pre_batch_does_not_block_post_batch(self) -> None:
        bridge = FakeBridge()
        store = FakeProcessedStore()
        pre_intents = [_intent(1, stake=5.0)]
        post_intents = [_intent(1, stake=5.0)]
        pre_intents[0] = replace(pre_intents[0], execution_phase="PRE", selection_id="runner-1")
        post_intents[0] = replace(post_intents[0], execution_phase="POST", selection_id="runner-1")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            pre_results, pre_skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=pre_intents,
                context=_context(),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            post_results, post_skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=post_intents,
                context=_context(),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        self.assertFalse(pre_skipped)
        self.assertFalse(post_skipped)
        self.assertEqual(pre_results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(post_results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(
            store.mark_calls,
            [
                ("parent:1|course:1|win:win-1|place:place-1|PRE", "win-1", "place-1"),
                ("parent:1|course:1|win:win-1|place:place-1|POST", "win-1", "place-1"),
            ],
        )

    def test_post_processed_key_is_unique_for_two_courses_same_parent(self) -> None:
        store = FakeProcessedStore()
        first_context = replace(
            _context(),
            course="Geelong (AUS)-202606160707",
            win_market_id="259181269",
            place_market_id="259181270",
        )
        second_context = replace(
            _context(),
            course="Geelong (AUS)-202606160726",
            win_market_id="259181272",
            place_market_id="259181273",
        )
        first_intent = replace(
            _intent(2, stake=2.0),
            execution_phase="POST",
            market_id="259181270",
            selection_id="2",
        )
        second_intent = replace(
            _intent(2, stake=2.0),
            execution_phase="POST",
            market_id="259181273",
            selection_id="2",
        )
        first_key = watch_gruss_real_strategy_test._processed_intent_key(
            "parent:35721835",
            first_intent,
            context=first_context,
            win_market_id="259181269",
            place_market_id="259181270",
        )
        second_key = watch_gruss_real_strategy_test._processed_intent_key(
            "parent:35721835",
            second_intent,
            context=second_context,
            win_market_id="259181272",
            place_market_id="259181273",
        )

        store.mark_seen(first_key, "259181269", "259181270")

        self.assertNotEqual(first_key, second_key)
        self.assertFalse(store.has_seen(second_key))
        self.assertIn("course:Geelong (AUS)-202606160707", first_key)
        self.assertIn("win:259181272", second_key)
        self.assertIn("place:259181273", second_key)

    def test_force_stake_two_is_logged_as_original_and_used(self) -> None:
        bridge = FakeBridge()

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_intent(1, stake=5.0)],
                context=_context(),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].stake_original, 5.0)
        self.assertEqual(results[0].stake_used, 2.0)
        self.assertTrue(results[0].stake_forced)
        self.assertEqual(rows[0]["stake_original"], "5.0")
        self.assertEqual(rows[0]["stake_used"], "2.0")
        self.assertEqual(rows[0]["stake_forced"], "True")

    def test_variable_stakes_keep_strategy_stake_and_provider_caps_at_five(self) -> None:
        bridge = FakeBridge()
        variable_env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="5",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", variable_env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_intent(1, stake=8.0)],
                context=_context(),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=None,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].stake_original, 8.0)
        self.assertEqual(results[0].stake_used, 5.0)
        self.assertFalse(results[0].stake_forced)
        self.assertTrue(results[0].stake_capped)
        self.assertEqual(results[0].stake_cap_value, 5.0)
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.0), ("S5", 5.0), ("Q5", "BACK")))
        self.assertEqual(rows[0]["stake_original"], "8.0")
        self.assertEqual(rows[0]["stake_used"], "5.0")
        self.assertEqual(rows[0]["stake_forced"], "False")
        self.assertEqual(rows[0]["stake_capped"], "True")
        self.assertEqual(rows[0]["stake_cap_value"], "5.0")

    def test_variable_stakes_allow_pre_ladder_and_provider_caps_at_five(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        store = FakeProcessedStore()
        variable_env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="5",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", variable_env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", stake=8.0)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=store,
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=None,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(skipped)
        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[0].stake_original, 8.0)
        self.assertEqual(results[0].stake_used, 5.0)
        self.assertFalse(results[0].stake_forced)
        self.assertTrue(results[0].stake_capped)
        self.assertEqual(results[0].stake_cap_value, 5.0)
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.0), ("S5", 5.0), ("Q5", "BACK")))
        self.assertEqual(rows[0]["stake_forced"], "False")
        self.assertEqual(rows[0]["stake_capped"], "True")
        self.assertEqual(rows[0]["stake_cap_value"], "5.0")
        self.assertEqual(len(store.mark_calls), 1)

    def test_variable_stakes_allow_pre_ladder_cap_above_five_with_override(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        store = FakeProcessedStore()
        variable_env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="7",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
            DOGBOT_GRUSS_REAL_ALLOW_VARIABLE_STAKE_OVER_5="true",
            DOGBOT_GRUSS_REAL_VARIABLE_STAKE_HARD_CAP="10",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", variable_env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", stake=8.0)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=store,
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=None,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(skipped)
        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[0].stake_original, 8.0)
        self.assertEqual(results[0].stake_used, 7.0)
        self.assertFalse(results[0].stake_forced)
        self.assertTrue(results[0].stake_capped)
        self.assertEqual(results[0].stake_cap_value, 7.0)
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.0), ("S5", 7.0), ("Q5", "BACK")))
        self.assertEqual(rows[0]["stake_forced"], "False")
        self.assertEqual(rows[0]["stake_capped"], "True")
        self.assertEqual(rows[0]["stake_cap_value"], "7.0")
        self.assertEqual(len(store.mark_calls), 1)

    def test_variable_stakes_pre_ladder_refuses_above_hard_cap(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        unsafe_env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="20",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
            DOGBOT_GRUSS_REAL_ALLOW_VARIABLE_STAKE_OVER_5="true",
            DOGBOT_GRUSS_REAL_VARIABLE_STAKE_HARD_CAP="10",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", unsafe_env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", stake=8.0)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=None,
            )

        self.assertFalse(skipped)
        self.assertEqual(results[0].status, "REJECTED_REAL")
        self.assertIn("pre_ladder_real_variable_max_stake_exceeds_hard_cap", results[0].reason)
        self.assertEqual(bridge.write_calls, [])

    def test_rejected_pre_ladder_guard_does_not_mark_processed(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        store = FakeProcessedStore()
        unsafe_env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_STAKE="3",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="2",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", unsafe_env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", stake=2.0)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=store,
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(skipped)
        self.assertEqual(results[0].status, "REJECTED_REAL")
        self.assertIn("pre_ladder_real_requires_max_stake_eq_2", results[0].reason)
        self.assertEqual(bridge.write_calls, [])
        self.assertEqual(store.mark_calls, [])
        self.assertEqual(rows[0]["status"], "REJECTED_REAL")

    def test_post_write_verification_and_trigger_cleanup_are_preserved(self) -> None:
        bridge = FakeBridge()

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_intent(1, stake=5.0)],
                context=_context(),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        result = results[0]
        self.assertTrue(result.post_write_verified)
        self.assertEqual(result.post_write_odds_cell_address, "R5")
        self.assertEqual(result.post_write_stake_cell_address, "S5")
        self.assertEqual(result.post_write_trigger_cell_address, "Q5")
        self.assertTrue(result.trigger_clear_attempted)
        self.assertTrue(result.trigger_cleared)
        self.assertEqual(result.trigger_clear_delay_ms, 0)
        self.assertEqual(bridge.clear_calls, [("PLACE", ("Q5", "R5", "S5"), "Q", True)])
        self.assertIsNone(bridge.cells[("PLACE", "Q5")])
        self.assertIsNone(bridge.cells[("PLACE", "R5")])
        self.assertIsNone(bridge.cells[("PLACE", "S5")])
        self.assertTrue(result.command_cells_clear_attempted)
        self.assertTrue(result.command_cells_cleared)
        self.assertEqual(result.command_cells_clear_reason, "command_cells_cleared")
        self.assertEqual(result.command_cells_clear_addresses, "Q5;R5;S5")

    def test_pre_ladder_step_one_writes_initial_back_and_reads_bet_ref_after(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"

        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="2")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4")],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[0].ladder_step, "1/4")
        self.assertTrue(results[0].trigger_written)
        self.assertEqual(results[0].trigger_value_written, "BACK")
        self.assertEqual(results[0].bet_ref_before, "")
        self.assertEqual(results[0].bet_ref_after, "432000000005")
        self.assertTrue(results[0].pre_bet_ref_required)
        self.assertTrue(results[0].pre_bet_ref_confirmed)
        self.assertFalse(results[0].pre_bet_ref_missing)
        self.assertFalse(results[0].update_allowed)
        self.assertEqual(results[0].stake_used, 2.0)
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.0), ("S5", 2.0), ("Q5", "BACK")))
        self.assertEqual(rows[0]["ladder_step"], "1/4")
        self.assertEqual(rows[0]["trigger_written"], "True")
        self.assertEqual(rows[0]["bet_ref_after"], "432000000005")
        self.assertEqual(rows[0]["pre_bet_ref_required"], "True")
        self.assertEqual(rows[0]["pre_bet_ref_confirmed"], "True")
        self.assertEqual(rows[0]["pre_bet_ref_missing"], "False")
        self.assertEqual(rows[0]["matched_stake"], "0.0")
        self.assertEqual(rows[0]["countdown_authorization_reason"], "pre_ladder_valid_milestone")
        self.assertEqual(rows[0]["bet_ref_poll_attempts"], "1")
        self.assertEqual(rows[0]["active_ladder_bet_ref_stored"], "True")

    def test_pre_ladder_initial_polls_briefly_and_stores_bet_ref_by_ladder(self) -> None:
        bridge = FakeDelayedBetRefBridge(reads_before_ref=2)
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        intents = [
            _pre_ladder_intent("1/4", ladder_id=f"ladder-{trap}", trap=trap)
            for trap in (1, 2, 3)
        ]
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="3",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="5",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual([result.status for result in results], ["GRUSS_PRE_LADDER_WRITTEN"] * 3)
        self.assertEqual([result.bet_ref_after for result in results], ["432000000005", "432000000006", "432000000007"])
        self.assertEqual(
            {ladder_id: state.bet_ref for ladder_id, state in provider.active_pre_ladders.items()},
            {"ladder-1": "432000000005", "ladder-2": "432000000006", "ladder-3": "432000000007"},
        )
        self.assertEqual([row["bet_ref_poll_attempts"] for row in rows], ["3", "3", "3"])
        self.assertTrue(all(row["pre_bet_ref_confirmed"] == "True" for row in rows))
        self.assertTrue(all(row["pre_bet_ref_missing"] == "False" for row in rows))
        self.assertTrue(all(row["active_ladder_bet_ref_stored"] == "True" for row in rows))
        self.assertTrue(all(row["bet_ref_lookup_source"].startswith("excel_row_batch_poll:") for row in rows))
        self.assertEqual([row["batch_size"] for row in rows], ["3", "3", "3"])
        self.assertEqual([row["order_index_in_batch"] for row in rows], ["1", "2", "3"])
        self.assertTrue(all(row["batch_write_duration_ms"] != "" for row in rows))
        self.assertTrue(all(row["bet_ref_collection_duration_ms"] != "" for row in rows))
        self.assertEqual([row["runner_row"] for row in rows], ["5", "6", "7"])
        self.assertEqual([row["runner_order_in_sheet"] for row in rows], ["1", "2", "3"])
        self.assertEqual([row["total_runners_in_gruss_sheet"] for row in rows], ["10", "10", "10"])
        self.assertEqual([row["mapped_runners_count"] for row in rows], ["3", "3", "3"])
        self.assertEqual([row["unmapped_runners_count"] for row in rows], ["0", "0", "0"])
        self.assertEqual([row["mapped_selection_ids"] for row in rows], ["1|2|3", "1|2|3", "1|2|3"])
        self.assertEqual([row["mapped_excel_rows"] for row in rows], ["5|6|7", "5|6|7", "5|6|7"])
        self.assertTrue(all("PLACE!T5=432000000005" in row["bet_ref_row_t_dump"] for row in rows))
        self.assertTrue(all("PLACE!T6=432000000006" in row["bet_ref_row_t_dump"] for row in rows))
        self.assertTrue(all("PLACE!T7=432000000007" in row["bet_ref_row_t_dump"] for row in rows))
        self.assertTrue(all(row["bet_ref_diagnostic_hold_after_batch"] == "False" for row in rows))
        first_read_index = next(index for index, event in enumerate(bridge.events) if event[0] == "read")
        self.assertEqual([event[0] for event in bridge.events[:first_read_index]], ["write", "write", "write"])

    def test_pre_ladder_initial_finds_bet_ref_in_place_selections_sheet(self) -> None:
        selections_rows = [
            ["Selection", "Bet ref", "Bet type", "Odds", "Stake", "Result", "Matched"],
            ["1. Runner 1", "432100000001", "B", 3.0, 2.0, "", 0.0],
        ]
        bridge = FakeSelectionsBetRefBridge([selections_rows])
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_GRUSS_BET_REF_LOOKUP_SOURCES="ROW_T,SELECTIONS_SHEET",
            DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_HOLD_AFTER_BATCH="true",
            DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS="7",
            DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS="7",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1)],
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "T5")] = None
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            update = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        first = first_results[0]
        self.assertEqual(first.bet_ref_after, "432100000001")
        self.assertEqual(first.bet_ref_lookup_source_used, "SELECTIONS_SHEET")
        self.assertTrue(first.active_ladder_bet_ref_stored)
        self.assertEqual(update.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(update.bet_ref_before, "432100000001")
        self.assertEqual(update.trigger_value_written, "BACKR")
        self.assertEqual(rows[0]["bet_ref_lookup_sources"], "ROW_T,SELECTIONS_SHEET")
        self.assertEqual(rows[0]["bet_ref_lookup_source_used"], "SELECTIONS_SHEET")
        self.assertEqual(rows[0]["row_t_value"], "")
        self.assertEqual(rows[0]["selections_match_found"], "True")
        self.assertEqual(rows[0]["selections_runner"], "1. Runner 1")
        self.assertEqual(rows[0]["selections_side"], "BACK")
        self.assertEqual(rows[0]["selections_stake"], "2.0")
        self.assertEqual(rows[0]["selections_bet_ref"], "432100000001")
        self.assertEqual(rows[0]["bet_ref_diagnostic_hold_after_batch"], "True")
        self.assertEqual(rows[0]["trigger_clear_delay_ms"], "7")
        self.assertIn("A=Selection", rows[0]["selections_sheet_headers"])
        self.assertIn("432100000001", rows[0]["selections_full_recent_rows"])
        self.assertEqual(rows[0]["workbook_sheet_names"], "WIN,PLACE,WIN_Selections,PLACE_Selections")
        self.assertIn("PLACE!row5:", rows[0]["runner_qz_dump"])
        self.assertIn("R5=", rows[0]["runner_qz_dump"])
        self.assertIn("S5=", rows[0]["runner_qz_dump"])
        self.assertEqual(rows[0]["command_cells_clear_attempted"], "True")
        self.assertEqual(rows[0]["command_cells_cleared"], "True")
        self.assertEqual(rows[0]["command_cells_clear_addresses"], "Q5;R5;S5")
        self.assertEqual(bridge.read_sheet_calls[0], ("PLACE_Selections", 1000, 80))

    def test_pre_ladder_initial_maps_multiple_selection_sheet_bet_refs_to_correct_runners(self) -> None:
        selections_rows = [
            ["Selection", "Bet ref", "Bet type", "Odds", "Stake", "Result", "Matched"],
            ["1. Runner 1", "432100000001", "B", 3.0, 2.0, "", 0.0],
            ["2. Runner 2", "432100000002", "B", 3.0, 2.0, "", 0.0],
            ["3. Runner 3", "432100000003", "B", 3.0, 2.0, "", 0.0],
        ]
        bridge = FakeSelectionsBetRefBridge([selections_rows])
        intents = [
            _pre_ladder_intent("1/4", ladder_id=f"ladder-{trap}", trap=trap)
            for trap in (1, 2, 3)
        ]
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="6",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "T5")] = None
            bridge.cells[("PLACE", "T6")] = None
            bridge.cells[("PLACE", "T7")] = None
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            update = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-2", trap=2),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )

        self.assertEqual([result.bet_ref_after for result in results], ["432100000001", "432100000002", "432100000003"])
        self.assertEqual(
            {ladder_id: state.bet_ref for ladder_id, state in provider.active_pre_ladders.items()},
            {"ladder-1": "432100000001", "ladder-2": "432000000006N", "ladder-3": "432100000003"},
        )
        self.assertEqual(update.bet_ref_before, "432100000002")
        self.assertEqual([call[1][2] for call in bridge.write_calls], [("Q5", "BACK"), ("Q6", "BACK"), ("Q7", "BACK"), ("Q6", "BACKR")])

    def test_pre_ladder_selection_sheet_does_not_assign_wrong_runner_bet_ref(self) -> None:
        selections_rows = [
            ["Selection", "Bet ref", "Bet type", "Odds", "Stake", "Result", "Matched"],
            ["2. Runner 2", "432100000002", "B", 3.0, 2.0, "", 0.0],
        ]
        bridge = FakeSelectionsBetRefBridge([selections_rows])
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1)],
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            update = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )

        self.assertEqual(first_results[0].bet_ref_after, "")
        self.assertFalse(first_results[0].active_ladder_bet_ref_stored)
        self.assertEqual(update.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(update.reason, "pre_ladder_retry_initial_written")
        self.assertTrue(update.pre_retry_allowed)
        self.assertEqual(update.pre_retry_reason, "missing_bet_ref_retry_initial_at_next_pre_step")
        self.assertFalse(update.update_allowed)
        self.assertEqual([call[1][2] for call in bridge.write_calls], [("Q5", "BACK"), ("Q5", "BACK")])

    def test_pre_ladder_missing_bet_ref_retry_does_not_stack_if_bet_ref_appears(self) -> None:
        bridge = FakeBetRefBridge(write_bet_ref_after_initial=False)
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            first_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            bridge.cells[("PLACE", "T5")] = "432123456789"
            update = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )

        self.assertEqual(first_results[0].status, "PRE_LADDER_BET_REF_MISSING_RETRYABLE")
        self.assertTrue(first_results[0].pre_retry_allowed)
        self.assertEqual(update.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(update.reason, "pre_ladder_replace_written")
        self.assertEqual(update.bet_ref_before, "432123456789")
        self.assertTrue(update.pre_bet_ref_late_detected)
        self.assertEqual(update.pre_bet_ref_late_value, "432123456789")
        self.assertFalse(update.pre_retry_allowed)
        self.assertEqual([call[1][2] for call in bridge.write_calls], [("Q5", "BACK"), ("Q5", "BACKR")])

    def test_pre_ladder_missing_bet_ref_with_matched_stake_blocks_retry(self) -> None:
        bridge = FakeMatchedNoBetRefBridge()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            first_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            retry = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        first = first_results[0]
        self.assertEqual(first.status, "PRE_LADDER_BET_REF_LATE_OR_MATCH_EVIDENCE")
        self.assertEqual(first.reason, "matched_evidence_found")
        self.assertTrue(first.matched_evidence_found)
        self.assertTrue(first.pending_ladder_created)
        self.assertFalse(first.pre_retry_allowed)
        self.assertIn("ladder-1", provider.active_pre_ladders)
        self.assertTrue(provider.active_pre_ladders["ladder-1"].pending_confirmation)
        self.assertEqual(retry.status, "GRUSS_PRE_LADDER_REPLACE_SKIPPED")
        self.assertEqual(retry.reason, "bet_ref_not_ready")
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(rows[0]["status"], "PRE_LADDER_BET_REF_LATE_OR_MATCH_EVIDENCE")
        self.assertEqual(rows[0]["matched_evidence_found"], "True")
        self.assertEqual(rows[0]["pending_ladder_created"], "True")
        self.assertEqual(rows[0]["pre_retry_allowed"], "False")
        self.assertEqual(rows[0]["pre_retry_block_reason"], "matched_evidence_found")

    def test_pre_ladder_missing_bet_ref_with_selection_evidence_blocks_retry(self) -> None:
        selections_rows = [
            [
                "Selection",
                "Bet ref",
                "Bet type",
                "Amount",
                "Average Odds",
                "Result",
                "Market Name",
                "Req Odds",
                "Matched Stake",
            ],
            ["1. Runner 1", "", "B", 2.0, 2.9, "", "Trap Challenge To Be Placed", 3.0, 1.25],
        ]
        bridge = FakeSelectionsBetRefBridge([selections_rows])
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_GRUSS_BET_REF_LOOKUP_SOURCES="SELECTIONS_SHEET",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1)],
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            retry = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        first = first_results[0]
        self.assertEqual(first.status, "PRE_LADDER_BET_REF_LATE_OR_MATCH_EVIDENCE")
        self.assertEqual(first.reason, "selection_row_evidence_found")
        self.assertTrue(first.selection_row_evidence_found)
        self.assertTrue(first.pending_ladder_created)
        self.assertFalse(first.pre_retry_allowed)
        self.assertTrue(provider.active_pre_ladders["ladder-1"].pending_confirmation)
        self.assertEqual(retry.status, "GRUSS_PRE_LADDER_REPLACE_SKIPPED")
        self.assertEqual(retry.reason, "bet_ref_not_ready")
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(rows[0]["selection_row_evidence_found"], "True")
        self.assertEqual(rows[0]["no_stacking_blocked_retry"], "True")
        self.assertEqual(rows[0]["pre_retry_block_reason"], "selection_row_evidence_found")

    def test_pre_ladder_missing_bet_ref_allows_at_most_two_retries(self) -> None:
        bridge = FakeBetRefBridge(write_bet_ref_after_initial=False)
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="10",
            DOGBOT_PRE_LADDER_BET_REF_MISSING_MAX_RETRIES="2",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            statuses = []
            reasons = []
            retry_counts = []
            for step, countdown in (("1/4", 45), ("2/4", 32), ("3/4", 20), ("4/4", 14)):
                bridge.cells[("PLACE", "D2")] = f"00:00:{countdown:02d}"
                results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                    provider=provider,
                    intents=[_pre_ladder_intent(step, ladder_id="ladder-1", trap=1)],
                    context=replace(_context(), countdown_seconds=countdown, milestone_seen=countdown),
                    processed_store=FakeProcessedStore(),
                    key=f"parent:1|milestone={countdown}|phase=PRE",
                    win_market_id="win-1",
                    place_market_id="place-1",
                    force_stake=2.0,
                )
                statuses.append(results[0].status)
                reasons.append(results[0].reason)
                retry_counts.append(results[0].pre_retry_count)
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(
            statuses,
            [
                "PRE_LADDER_BET_REF_MISSING_RETRYABLE",
                "PRE_LADDER_BET_REF_MISSING_RETRYABLE",
                "PRE_LADDER_BET_REF_MISSING",
                "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
            ],
        )
        self.assertEqual(reasons[-2:], ["pre_bet_ref_missing_retry_limit_reached"] * 2)
        self.assertEqual(retry_counts, [0, 1, 2, 2])
        self.assertEqual(len(bridge.write_calls), 3)
        self.assertEqual([call[1][2] for call in bridge.write_calls], [("Q5", "BACK"), ("Q5", "BACK"), ("Q5", "BACK")])
        self.assertEqual([row["pre_retry_count"] for row in rows], ["0", "1", "2", "2"])
        self.assertEqual(rows[2]["pre_retry_allowed"], "False")
        self.assertEqual(rows[2]["pre_retry_block_reason"], "pre_bet_ref_missing_retry_limit_reached")
        self.assertEqual(rows[3]["update_skipped_reason"], "pre_bet_ref_missing_retry_limit_reached")

    def test_pre_ladder_retry_preserves_value_clamp_and_floor_price(self) -> None:
        bridge = FakeBetRefBridge(write_bet_ref_after_initial=False)
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="3",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            retry_intent = replace(
                _pre_ladder_intent("2/4", ladder_id="ladder-1", trap=1),
                provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                dry_run=False,
                price=1.01,
                computed_limit_price_effective=1.01,
                pre_value_target_price=1.01,
                sent_price_before_value_clamp=0.5,
                value_clamp_applied=True,
                min_price_floor_applied=True,
                stake=2.0,
                stake_original=5.0,
                stake_forced=True,
            )
            retry = provider.place_order(
                retry_intent,
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(retry.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertTrue(retry.pre_retry_allowed)
        self.assertEqual(bridge.write_calls[1][1][0], ("R5", 1.01))
        self.assertEqual(rows[1]["computed_limit_price_effective"], "1.01")
        self.assertEqual(rows[1]["sent_price_after_value_clamp"], "1.01")
        self.assertEqual(rows[1]["min_price_floor_applied"], "True")

    def test_pre_ladder_selection_sheet_matches_dirty_name_side_l_and_close_req_odds(self) -> None:
        selections_rows = [
            [
                "Selection name",
                "Bet Ref",
                "Bet type",
                "Amount",
                "Average Odds",
                "Result",
                "Market Name",
                "Req Odds",
                "Req Stake",
                "Matched Odds",
                "Matched Stake",
            ],
            ["1. Runner's   Hero", "432200000001", "L", 2.0, "", "", "Trap Challenge To Be Placed", 3.05, 2.01, "", 0.0],
        ]
        bridge = FakeSelectionsBetRefBridge([selections_rows])
        intent = _pre_ladder_intent("1/4", ladder_id="ladder-lay-1", trap=1, side="LAY")
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_GRUSS_BET_REF_LOOKUP_SOURCES="SELECTIONS_SHEET",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[intent],
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].bet_ref_after, "432200000001")
        self.assertEqual(results[0].selections_side, "LAY")
        self.assertEqual(results[0].selections_req_odds, 3.05)
        self.assertIn("trap_match", results[0].selections_match_reason)
        self.assertIn("req_odds_match", results[0].selections_match_reason)
        self.assertIn("Trap Challenge To Be Placed", rows[0]["selections_debug_recent_rows"])
        self.assertIn("score=", rows[0]["selections_top_candidates"])

    def test_pre_ladder_selection_sheet_logs_top_candidates_when_unmatched(self) -> None:
        selections_rows = [
            [
                "Selection name",
                "Bet Ref",
                "Bet type",
                "Amount",
                "Average Odds",
                "Result",
                "Market Name",
                "Req Odds",
                "Req Stake",
                "Matched Odds",
                "Matched Stake",
            ],
            ["Wrong Runner", "SEL-WRONG", "B", 2.0, "", "", "Dunstall Park 11th Jun - 18:18 To Be Placed", 3.0, 2.0, "", 0.0],
            ["Runner 1", "SEL-WIN", "B", 2.0, "", "", "Win Market", 3.0, 2.0, "", 0.0],
        ]
        bridge = FakeSelectionsBetRefBridge([selections_rows])
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_GRUSS_BET_REF_LOOKUP_SOURCES="SELECTIONS_SHEET",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            processed_store = FakeProcessedStore()
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1)],
                context=replace(_context(), countdown_seconds=20, course="Dunstall Park 11th Jun"),
                processed_store=processed_store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "PRE_LADDER_BET_REF_MISSING_RETRYABLE")
        self.assertEqual(results[0].reason, "pre_ladder_bet_ref_missing_retryable_after_poll")
        self.assertTrue(results[0].pre_bet_ref_required)
        self.assertFalse(results[0].pre_bet_ref_confirmed)
        self.assertTrue(results[0].pre_bet_ref_missing)
        self.assertEqual(results[0].pre_unconfirmed_reason, "bet_ref_missing_after_poll")
        self.assertEqual(results[0].bet_ref_after, "")
        self.assertFalse(results[0].active_ladder_bet_ref_stored)
        self.assertNotIn("ladder-1", provider.active_pre_ladders)
        self.assertEqual(processed_store.mark_calls, [])
        self.assertIn("no_matching_selection_row", results[0].selections_match_reason)
        self.assertEqual(rows[0]["status"], "PRE_LADDER_BET_REF_MISSING_RETRYABLE")
        self.assertEqual(rows[0]["pre_bet_ref_missing"], "True")
        self.assertEqual(rows[0]["pre_bet_ref_missing_retryable"], "True")
        self.assertEqual(rows[0]["pre_retry_allowed"], "True")
        self.assertIn("SEL-WRONG", rows[0]["selections_top_candidates"])
        self.assertIn("SEL-WIN", rows[0]["selections_top_candidates"])
        self.assertIn("Win Market", rows[0]["selections_debug_recent_rows"])
        self.assertEqual(rows[0]["selections_market_query"], "Dunstall Park 11th Jun")
        self.assertIn("SEL-WRONG", rows[0]["selections_current_market_rows"])
        self.assertIn("SEL-WIN", rows[0]["selections_debug_recent_rows"])

    def test_pre_ladder_diagnostic_keep_triggers_leaves_initial_trigger_visible(self) -> None:
        bridge = FakeBetRefBridge(write_bet_ref_after_initial=False)
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="1",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
            DOGBOT_GRUSS_DIAGNOSTIC_KEEP_TRIGGERS="true",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="1",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1)],
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "PRE_LADDER_BET_REF_MISSING_RETRYABLE")
        self.assertEqual(bridge.cells[("PLACE", "Q5")], "BACK")
        self.assertEqual(bridge.clear_calls, [])
        self.assertEqual(rows[0]["trigger_clear_attempted"], "False")
        self.assertEqual(rows[0]["trigger_clear_reason"], "diagnostic_keep_triggers_enabled")
        self.assertEqual(rows[0]["diagnostic_keep_triggers"], "True")
        self.assertEqual(rows[0]["pre_bet_ref_missing"], "True")

    def test_pre_ladder_following_step_uses_backr_only_when_bet_ref_exists(self) -> None:
        bridge = FakeBetRefBridge()
        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="2")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            first_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4")],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("2/4")],
                context=replace(_context(), countdown_seconds=32, milestone_seen=32),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=32|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(first_results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[0].trigger_value_written, "BACKR")
        self.assertEqual(results[0].bet_ref_before, "432000000005")
        self.assertEqual(results[0].bet_ref_after, "432000000005N")
        self.assertTrue(results[0].update_allowed)
        self.assertTrue(results[0].replace_allowed)
        self.assertEqual(results[0].replace_trigger, "BACKR")
        self.assertEqual(bridge.write_calls[1][1], (("R5", 3.0), ("S5", 2.0), ("Q5", "BACKR")))
        self.assertEqual(rows[1]["update_allowed"], "True")
        self.assertEqual(rows[1]["replace_allowed"], "True")
        self.assertEqual(rows[1]["replace_trigger"], "BACKR")
        self.assertEqual(rows[1]["matched_stake_cell_address"], "W5")
        self.assertEqual(rows[1]["bet_ref_before"], "432000000005")
        self.assertEqual(rows[1]["bet_ref_after"], "432000000005N")

    def test_pre_ladder_replace_uses_stored_bet_ref_for_the_correct_runner(self) -> None:
        bridge = FakeDelayedBetRefBridge(reads_before_ref=0)
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="3",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
            DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS="2",
            DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS="0",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            first_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4", ladder_id="ladder-2", trap=2)],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            first = first_results[0]
            bridge.cells[("PLACE", "T6")] = None
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            update = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-2", trap=2),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )
            bridge.cells[("PLACE", "D2")] = "00:00:20"
            wrong_runner = provider.place_order(
                replace(
                    _pre_ladder_intent("3/4", ladder_id="ladder-2", trap=3),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=20, milestone_seen=20),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(first.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(first.bet_ref_after, "432000000006")
        self.assertEqual(update.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(update.bet_ref_before, "432000000006")
        self.assertEqual(update.bet_ref_lookup_source, "active_ladder_state")
        self.assertEqual(update.trigger_value_written, "BACKR")
        self.assertEqual(wrong_runner.status, "GRUSS_PRE_LADDER_REPLACE_SKIPPED")
        self.assertEqual(wrong_runner.reason, "active_ladder_runner_mismatch_do_not_replace")
        self.assertEqual([call[1][2] for call in bridge.write_calls], [("Q6", "BACK"), ("Q6", "BACKR")])
        self.assertEqual(rows[1]["bet_ref_lookup_source"], "active_ladder_state")
        self.assertEqual(rows[1]["bet_ref_before"], "432000000006")
        self.assertEqual(rows[2]["bet_ref_lookup_source"], "active_ladder_state_mismatch")

    def test_pre_ladder_replace_waits_for_pendingr_and_updates_active_bet_ref(self) -> None:
        bridge = FakePendingReplaceBetRefBridge(reads_before_replace_ref=1)
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_GRUSS_REPLACE_BET_REF_WAIT_MS="200",
            DOGBOT_GRUSS_REPLACE_BET_REF_POLL_MS="10",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            first_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4")],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("2/4")],
                context=replace(_context(), countdown_seconds=32, milestone_seen=32),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=32|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(first_results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[0].bet_ref_after, "432900000005N")
        self.assertTrue(results[0].active_ladder_bet_ref_updated)
        self.assertEqual(provider.active_pre_ladders["ladder-1"].bet_ref, "432900000005N")
        self.assertEqual(rows[1]["replace_bet_ref_wait_attempted"], "True")
        self.assertEqual(rows[1]["replace_bet_ref_wait_result"], "resolved")
        self.assertEqual(rows[1]["bet_ref_before_wait"], "PENDINGR")
        self.assertEqual(rows[1]["bet_ref_after_wait"], "432900000005N")
        self.assertEqual(rows[1]["active_ladder_bet_ref_updated"], "True")

    def test_pre_ladder_following_step_skips_when_pendingr_timeout_persists(self) -> None:
        bridge = FakePendingReplaceBetRefBridge(reads_before_replace_ref=None)
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="3",
            DOGBOT_GRUSS_REPLACE_BET_REF_WAIT_MS="20",
            DOGBOT_GRUSS_REPLACE_BET_REF_POLL_MS="10",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4")],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            step2_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("2/4")],
                context=replace(_context(), countdown_seconds=32, milestone_seen=32),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=32|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:20"
            step3_results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("3/4")],
                context=replace(_context(), countdown_seconds=20, milestone_seen=20),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=20|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(step2_results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(step2_results[0].replace_bet_ref_wait_result, "timeout")
        self.assertEqual(step2_results[0].bet_ref_after, "PENDINGR")
        self.assertFalse(step2_results[0].active_ladder_bet_ref_updated)
        self.assertEqual(step3_results[0].status, "GRUSS_PRE_LADDER_REPLACE_SKIPPED")
        self.assertEqual(step3_results[0].reason, "replace_skipped_bet_ref_still_pending")
        self.assertEqual(step3_results[0].bet_ref_status_value, "PENDINGR")
        self.assertEqual(len(bridge.write_calls), 2)
        self.assertEqual(rows[1]["replace_bet_ref_wait_result"], "timeout")
        self.assertEqual(rows[2]["replace_bet_ref_wait_attempted"], "True")
        self.assertEqual(rows[2]["replace_bet_ref_wait_result"], "timeout")
        self.assertEqual(rows[2]["update_skipped_reason"], "replace_skipped_bet_ref_still_pending")
        self.assertEqual(rows[2]["replace_skipped_bet_ref_still_pending"], "True")

    def test_pre_ladder_following_step_without_bet_ref_does_not_write_or_stack(self) -> None:
        bridge = FakeBetRefBridge(write_bet_ref_after_initial=False)
        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="2")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4")],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("2/4")],
                context=replace(_context(), countdown_seconds=32, milestone_seen=32),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=32|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "PRE_LADDER_BET_REF_MISSING_RETRYABLE")
        self.assertEqual(results[0].reason, "pre_ladder_bet_ref_missing_retryable_after_poll")
        self.assertTrue(results[0].pre_retry_allowed)
        self.assertEqual(results[0].pre_retry_reason, "bet_ref_missing_retry_next_pre_step")
        self.assertFalse(results[0].update_allowed)
        self.assertEqual(len(bridge.write_calls), 2)
        self.assertEqual([call[1][2] for call in bridge.write_calls], [("Q5", "BACK"), ("Q5", "BACK")])
        self.assertEqual(rows[1]["status"], "PRE_LADDER_BET_REF_MISSING_RETRYABLE")
        self.assertEqual(rows[1]["pre_retry_allowed"], "True")

    def test_pre_ladder_replace_refuses_non_replaceable_row_states(self) -> None:
        cases = [
            ("PENDING", 0.0, "bet_ref_not_ready"),
            ("LAPSED", 0.0, "row_status_not_replaceable"),
            ("RESULT_LOST", 0.0, "row_status_not_replaceable"),
            ("BR-5", 0.0, "invalid_bet_ref_for_replace"),
            ("432000000005", "", "matched_stake_unavailable_no_replace"),
            ("432000000005", 0.5, "matched_stake_positive_no_replace"),
        ]
        for bet_ref, matched_stake, expected_reason in cases:
            with self.subTest(bet_ref=bet_ref, matched_stake=matched_stake):
                bridge = FakeBetRefBridge()
                env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="2")
                with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
                    provider = GrussExcelOrderProvider(tmp, bridge=bridge)
                    bridge.cells[("PLACE", "D2")] = "00:00:45"
                    watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                        provider=provider,
                        intents=[_pre_ladder_intent("1/4")],
                        context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                        processed_store=FakeProcessedStore(),
                        key="parent:1|milestone=45|phase=PRE",
                        win_market_id="win-1",
                        place_market_id="place-1",
                        force_stake=2.0,
                    )
                    bridge.cells[("PLACE", "T5")] = bet_ref
                    bridge.cells[("PLACE", "W5")] = matched_stake
                    provider.active_pre_ladders["ladder-1"].bet_ref = bet_ref
                    bridge.cells[("PLACE", "D2")] = "00:00:32"
                    results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                        provider=provider,
                        intents=[_pre_ladder_intent("2/4")],
                        context=replace(_context(), countdown_seconds=32, milestone_seen=32),
                        processed_store=FakeProcessedStore(),
                        key="parent:1|milestone=32|phase=PRE",
                        win_market_id="win-1",
                        place_market_id="place-1",
                        force_stake=2.0,
                    )

                self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_REPLACE_SKIPPED")
                self.assertEqual(results[0].reason, expected_reason)
                self.assertEqual(len(bridge.write_calls), 1)

    def test_pre_ladder_replace_refuses_unavailable_countdown(self) -> None:
        bridge = FakeBetRefBridge()
        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="2")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4")],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "T5")] = "432000000005"
            bridge.cells[("PLACE", "W5")] = 0.0
            bridge.cells[("PLACE", "D2")] = ""
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("2/4")],
                context=replace(_context(), countdown_seconds=32, milestone_seen=32),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=32|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_REPLACE_SKIPPED")
        self.assertEqual(results[0].reason, "countdown_unavailable_no_replace")
        self.assertEqual(len(bridge.write_calls), 1)

    def test_pre_ladder_replace_refuses_countdown_lte_ten(self) -> None:
        bridge = FakeBetRefBridge()
        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="2")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("1/4")],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=45|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            bridge.cells[("PLACE", "T5")] = "432000000005"
            bridge.cells[("PLACE", "W5")] = 0.0
            bridge.cells[("PLACE", "D2")] = "00:00:10"
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_pre_ladder_intent("2/4")],
                context=replace(_context(), countdown_seconds=32, milestone_seen=32),
                processed_store=FakeProcessedStore(),
                key="parent:1|milestone=32|phase=PRE",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_REPLACE_SKIPPED")
        self.assertEqual(results[0].reason, "countdown_too_low_no_replace")
        self.assertEqual(results[0].countdown_at_write, 10)
        self.assertEqual(len(bridge.write_calls), 1)

    def test_pre_ladder_allows_only_one_active_ladder(self) -> None:
        bridge = FakeBetRefBridge()
        intents = [
            _pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1),
            _pre_ladder_intent("1/4", ladder_id="ladder-2", trap=2),
        ]
        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="2")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(results[1].status, "REJECTED_REAL")
        self.assertEqual(results[1].reason, "max_active_pre_ladder_reached")
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(rows[1]["update_skipped_reason"], "max_active_pre_ladder_reached")

    def test_pre_ladder_max_ladders_three_allows_three_distinct_ladders(self) -> None:
        bridge = FakeBetRefBridge()
        intents = [
            _pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1),
            _pre_ladder_intent("1/4", ladder_id="ladder-2", trap=2),
            _pre_ladder_intent("1/4", ladder_id="ladder-3", trap=3),
        ]
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="3",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="3",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual([result.status for result in results], ["GRUSS_PRE_LADDER_WRITTEN"] * 3)
        self.assertEqual(len(bridge.write_calls), 3)
        self.assertEqual(set(provider.active_pre_ladders), {"ladder-1", "ladder-2", "ladder-3"})
        self.assertTrue(all(row["reason"] != LEGACY_MAX_LADDERS_REASON for row in rows))
        self.assertEqual(rows[-1]["max_active_pre_ladders"], "3")
        self.assertEqual(rows[-1]["active_pre_ladder_count"], "3")

    def test_pre_ladder_max_ladders_hundred_allows_multiple_distinct_ladders(self) -> None:
        bridge = FakeBetRefBridge()
        intents = [
            _pre_ladder_intent("1/4", ladder_id=f"ladder-{index}", trap=index)
            for index in range(1, 6)
        ]
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="500",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=20),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual([result.status for result in results], ["GRUSS_PRE_LADDER_WRITTEN"] * 5)
        self.assertEqual(len(bridge.write_calls), 5)
        self.assertTrue(all(LEGACY_MAX_LADDERS_REASON not in result.reason for result in results))
        self.assertTrue(all(row["reason"] != LEGACY_MAX_LADDERS_REASON for row in rows))
        self.assertEqual(rows[-1]["max_active_pre_ladders"], "100")
        self.assertEqual(rows[-1]["active_pre_ladder_count"], "5")

    def test_pre_ladder_variable_stakes_with_max_ladders_fifty_writes_multiple_ladders(self) -> None:
        bridge = FakeBetRefBridge()
        stakes = [2.0, 2.36, 2.97, 3.63, 3.73, 6.2]
        intents = [
            _pre_ladder_intent(
                "1/4",
                ladder_id=f"ladder-{index}",
                trap=index,
                stake=stake,
            )
            for index, stake in enumerate(stakes, start=1)
        ]
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_VARIABLE_STAKES="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="5",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="",
            DOGBOT_GRUSS_REAL_MAX_ORDERS="500",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="50",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            bridge.cells[("PLACE", "D2")] = "00:00:45"
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=replace(_context(), countdown_seconds=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=None,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual([result.status for result in results], ["GRUSS_PRE_LADDER_WRITTEN"] * 6)
        self.assertTrue(all(result.reason != "max_active_pre_ladder_reached" for result in results))
        self.assertEqual(len(bridge.write_calls), 6)
        written_stakes = [float(call[1][1][1]) for call in bridge.write_calls]
        self.assertEqual(written_stakes, [2.0, 2.36, 2.97, 3.63, 3.73, 5.0])
        self.assertTrue(all(stake <= 5.0 for stake in written_stakes))
        self.assertTrue(all(row["reason"] != LEGACY_MAX_LADDERS_REASON for row in rows))
        self.assertTrue(all(row["reason"] != "max_active_pre_ladder_reached" for row in rows))
        self.assertEqual(rows[-1]["max_ladders_limit"], "50")
        self.assertEqual(rows[-1]["active_ladder_count"], "6")
        self.assertTrue(all(row["stake_forced"] == "False" for row in rows))
        self.assertEqual(rows[-1]["stake_capped"], "True")
        self.assertEqual(rows[-1]["stake_cap_value"], "5.0")

    def test_pre_ladder_conflict_keeps_back_when_back_price_is_closer(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        back = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-back", trap=1, side="BACK"),
            price=5.7,
            best_same_side_lay_offer=5.6,
            best_same_side_back_offer=2.0,
        )
        lay = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-lay", trap=1, side="LAY"),
            price=1.06,
            best_same_side_lay_offer=5.6,
            best_same_side_back_offer=2.0,
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[back, lay],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(bridge.write_calls[0][1][-1], ("Q5", "BACK"))
        self.assertEqual([result.status for result in results], ["REJECTED_REAL", "GRUSS_PRE_LADDER_WRITTEN"])
        self.assertEqual([result.reason for result in results], ["conflicting_back_lay_lost_priority", "pre_ladder_step_written"])
        self.assertEqual(rows[0]["conflict_detected"], "True")
        self.assertEqual(rows[0]["conflict_type"], "back_lay_same_runner_market_phase")
        self.assertEqual(rows[0]["selected_side"], "BACK")
        self.assertEqual(rows[0]["rejected_side"], "LAY")
        self.assertEqual(rows[0]["back_systems"], "BACK_PLACE_101")
        self.assertEqual(rows[0]["lay_systems"], "LAY_PLACE_301")
        self.assertEqual(rows[0]["conflict_resolution_reason"], "per_runner_nearest_price")
        self.assertEqual(rows[0]["pre_back_lay_conflict"], "True")
        self.assertEqual(rows[0]["pre_conflict_chosen_side"], "BACK")
        self.assertEqual(rows[0]["pre_conflict_rejected_side"], "LAY")
        self.assertEqual(rows[0]["pre_conflict_reason"], "pre_conflict_back_nearer")

    def test_pre_ladder_conflict_keeps_lay_when_lay_price_is_closer(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        back = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-back", trap=1, side="BACK"),
            price=5.7,
            best_same_side_lay_offer=7.0,
            best_same_side_back_offer=2.2,
        )
        lay = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-lay", trap=1, side="LAY"),
            price=2.1,
            best_same_side_lay_offer=7.0,
            best_same_side_back_offer=2.2,
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[back, lay],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        self.assertEqual(bridge.write_calls[0][1][-1], ("Q5", "LAY"))
        self.assertEqual([result.status for result in results], ["REJECTED_REAL", "GRUSS_PRE_LADDER_WRITTEN"])
        self.assertEqual([result.reason for result in results], ["conflicting_back_lay_lost_priority", "pre_ladder_step_written"])
        self.assertEqual(results[0].selected_side, "LAY")
        self.assertEqual(results[0].rejected_side, "BACK")
        self.assertEqual(results[0].pre_conflict_reason, "pre_conflict_lay_nearer")

    def test_pre_ladder_conflict_uses_price_distance_before_edge(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        back = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-back", trap=1, side="BACK"),
            price=5.7,
            best_same_side_lay_offer=5.8,
            best_same_side_back_offer=2.2,
            strategy_edge=0.25,
        )
        lay = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-lay", trap=1, side="LAY"),
            price=2.1,
            best_same_side_lay_offer=5.8,
            best_same_side_back_offer=2.2,
            strategy_edge=-0.05,
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[back, lay],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(bridge.write_calls[0][1][-1], ("Q5", "BACK"))
        self.assertEqual([result.status for result in results], ["REJECTED_REAL", "GRUSS_PRE_LADDER_WRITTEN"])
        self.assertEqual([result.reason for result in results], ["conflicting_back_lay_lost_priority", "pre_ladder_step_written"])
        self.assertEqual(rows[0]["conflict_group_key"], "parent:1|place-1|PLACE|1|PRE")
        self.assertEqual(rows[0]["conflict_candidates_count"], "2")
        self.assertEqual(rows[0]["pre_conflict_chosen_side"], "BACK")
        self.assertEqual(rows[0]["pre_conflict_reason"], "pre_conflict_back_nearer")
        self.assertEqual(rows[0]["conflict_resolution_reason"], "per_runner_nearest_price")
        self.assertEqual(rows[0]["pre_conflict_group_key"], "parent:1|place-1|PLACE|1|PRE")
        self.assertEqual(rows[0]["pre_conflict_selection_id"], "1")

    def test_pre_ladder_conflict_is_resolved_per_runner_not_per_market(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="4",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        runner1_back = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-1-back", trap=1, side="BACK"),
            price=5.8,
            best_same_side_lay_offer=6.0,
            best_same_side_back_offer=2.0,
        )
        runner1_lay = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-1-lay", trap=1, side="LAY"),
            price=1.5,
            best_same_side_lay_offer=6.0,
            best_same_side_back_offer=2.0,
        )
        runner2_back = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-2-back", trap=2, side="BACK"),
            price=9.0,
            best_same_side_lay_offer=6.0,
            best_same_side_back_offer=2.2,
        )
        runner2_lay = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-2-lay", trap=2, side="LAY"),
            price=2.24,
            best_same_side_lay_offer=6.0,
            best_same_side_back_offer=2.2,
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[runner1_back, runner1_lay, runner2_back, runner2_lay],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        written = [result for result in results if result.status == "GRUSS_PRE_LADDER_WRITTEN"]
        rejected = [result for result in results if result.status == "REJECTED_REAL"]
        self.assertEqual(len(written), 2)
        self.assertEqual(len(rejected), 2)
        self.assertEqual([call[1][-1] for call in bridge.write_calls[:2]], [("Q5", "BACK"), ("Q6", "LAY")])
        self.assertEqual({result.pre_conflict_selection_id for result in written}, {"1", "2"})
        self.assertEqual({row["pre_conflict_group_key"] for row in rows if row["pre_back_lay_conflict"] == "True"}, {
            "parent:1|place-1|PLACE|1|PRE",
            "parent:1|place-1|PLACE|2|PRE",
        })

    def test_pre_ladder_conflict_uses_stable_tiebreak_when_price_distance_equal(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        back = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-back", trap=1, side="BACK"),
            price=6.0,
            best_same_side_lay_offer=5.0,
            best_same_side_back_offer=5.0,
        )
        lay = replace(
            _pre_ladder_intent("1/4", ladder_id="ladder-lay", trap=1, side="LAY"),
            price=4.0,
            best_same_side_lay_offer=5.0,
            best_same_side_back_offer=5.0,
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[back, lay],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(bridge.write_calls[0][1][-1], ("Q5", "BACK"))
        written = next(result for result in results if result.status == "GRUSS_PRE_LADDER_WRITTEN")
        rejected = next(result for result in results if result.status == "REJECTED_REAL")
        self.assertEqual(written.trigger_value_written, "BACK")
        self.assertEqual(written.reason, "pre_ladder_step_written")
        self.assertEqual(rejected.reason, "conflicting_back_lay_lost_priority")
        conflict_row = next(row for row in rows if row["pre_conflict_reason"] == "equal_distance_tiebreak_back")
        self.assertEqual(conflict_row["conflict_resolution_reason"], "per_runner_nearest_price")
        self.assertEqual(conflict_row["pre_conflict_chosen_side"], "BACK")
        self.assertEqual(conflict_row["pre_conflict_rejected_side"], "LAY")

    def test_pre_ladder_conflict_rejects_missing_reference(self) -> None:
        bridge = FakeBetRefBridge()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        back = replace(_pre_ladder_intent("1/4", ladder_id="ladder-back", trap=1, side="BACK"), price=6.0)
        lay = replace(_pre_ladder_intent("1/4", ladder_id="ladder-lay", trap=1, side="LAY"), price=4.0)

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[back, lay],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        self.assertEqual([result.status for result in results], ["REJECTED_REAL", "REJECTED_REAL"])
        self.assertEqual([result.reason for result in results], ["pre_conflict_missing_reference_no_bet"] * 2)
        self.assertEqual(results[0].pre_conflict_reason, "pre_conflict_missing_reference_no_bet")
        self.assertEqual(bridge.write_calls, [])

    def test_post_back_lay_conflict_rejects_both_without_provider_write(self) -> None:
        bridge = FakeBetRefBridge()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        back = replace(
            _intent(1),
            side="BACK",
            strategy_id="BACK_PLACE_206",
            runner_name="Lacy Sarah",
            execution_phase="POST",
            selection_id="1",
        )
        lay = replace(
            _intent(1),
            side="LAY",
            order_type="SP_MOC",
            strategy_id="LAY_PLACE_502",
            runner_name="Lacy Sarah",
            execution_phase="POST",
            selection_id="1",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[back, lay],
                context=replace(_context(), countdown_seconds=0, milestone_seen=0),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(bridge.write_calls, [])
        self.assertEqual([result.status for result in results], ["REJECTED_REAL", "REJECTED_REAL"])
        self.assertEqual([result.reason for result in results], ["conflicting_back_lay_no_bet"] * 2)
        self.assertEqual(rows[0]["conflict_detected"], "True")
        self.assertEqual(rows[0]["conflict_type"], "back_lay_same_runner_market_phase")
        self.assertEqual(rows[0]["selected_side"], "NONE")
        self.assertEqual(rows[0]["rejected_side"], "BOTH")
        self.assertEqual(rows[0]["back_systems"], "BACK_PLACE_206")
        self.assertEqual(rows[0]["lay_systems"], "LAY_PLACE_502")
        self.assertEqual(rows[0]["post_provider_called"], "False")
        self.assertEqual(rows[0]["post_write_attempted"], "False")

    def test_lay_alone_pre_reaches_gruss_provider(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = 45 / 86400
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        intent = _pre_ladder_intent("1/4", ladder_id="ladder-lay", trap=1, side="LAY", stake=2.0)

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[intent],
                context=replace(_context(), countdown_seconds=45, milestone_seen=45),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1][2], ("Q5", "LAY"))
        self.assertEqual(rows[0]["side"], "LAY")

    def test_lay_alone_post_reaches_gruss_provider(self) -> None:
        bridge = FakeBetRefBridge()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        intent = replace(
            _intent(1, stake=2.0),
            provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
            dry_run=False,
            side="LAY",
            strategy_id="LAY_PLACE_502",
            execution_phase="POST",
            selection_id="1",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[intent],
                context=replace(_context(), countdown_seconds=0, milestone_seen=0),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1][2], ("Q5", "LAY"))
        self.assertEqual(rows[0]["side"], "LAY")

    def test_pre_ladder_rejects_when_milestone_window_is_missed(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = 4 / 86400
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="1",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                    signal_countdown_seconds=20,
                    market_reference_price_at_signal=3.0,
                ),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "pre_ladder_milestone_window_missed")
        self.assertEqual(bridge.write_calls, [])
        self.assertEqual(rows[0]["countdown_at_write"], "4")

    def test_pre_ladder_rejects_stale_market_price_before_write(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = 20 / 86400
        bridge.cells[("PLACE", "O5")] = 5.2
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="1",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
            DOGBOT_PRE_LADDER_MAX_STALE_PRICE_DISTANCE_PCT="0.25",
            DOGBOT_PRE_IGNORE_STALE_PRICE_BEFORE_WRITE="false",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                    signal_countdown_seconds=20,
                    market_reference_price_at_signal=2.2,
                ),
                replace(_context(), countdown_seconds=20, milestone_seen=20),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "stale_market_price_before_write")
        self.assertEqual(bridge.write_calls, [])
        self.assertEqual(rows[0]["market_reference_price_at_signal"], "2.2")
        self.assertEqual(rows[0]["current_market_price_at_write"], "5.2")
        self.assertEqual(rows[0]["stale_check_ignored_for_pre"], "False")

    def test_pre_ladder_ignores_stale_market_price_when_configured(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = 20 / 86400
        bridge.cells[("PLACE", "O5")] = 5.2
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="1",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
            DOGBOT_PRE_LADDER_MAX_STALE_PRICE_DISTANCE_PCT="0.25",
            DOGBOT_PRE_IGNORE_STALE_PRICE_BEFORE_WRITE="true",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                    signal_countdown_seconds=20,
                    market_reference_price_at_signal=2.2,
                ),
                replace(_context(), countdown_seconds=20, milestone_seen=20),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(result.reason, "pre_ladder_step_written")
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(bridge.write_calls[0][1][2], ("Q5", "BACK"))
        self.assertEqual(rows[0]["market_reference_price_at_signal"], "2.2")
        self.assertEqual(rows[0]["current_market_price_at_write"], "5.2")
        self.assertNotEqual(rows[0]["stale_distance"], "")
        self.assertEqual(rows[0]["stale_price_limit"], "0.25")
        self.assertEqual(rows[0]["stale_check_ignored_for_pre"], "True")

    def test_pre_ladder_update_existing_ladder_does_not_consume_new_ladder_slot(self) -> None:
        bridge = FakeBetRefBridge()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="3",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="1",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=20, milestone_seen=20),
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            update = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )
            blocked_other = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4", ladder_id="ladder-2", trap=2),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=20, milestone_seen=20),
            )

        self.assertEqual(first.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(update.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertTrue(update.update_allowed)
        self.assertEqual(update.trigger_value_written, "BACKR")
        self.assertEqual(blocked_other.status, "REJECTED_REAL")
        self.assertEqual(blocked_other.reason, "max_active_pre_ladder_reached")
        self.assertEqual(set(provider.active_pre_ladders), {"ladder-1"})

    def test_pre_ladder_replaces_count_as_real_order_writes(self) -> None:
        bridge = FakeBetRefBridge()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="1",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=20, milestone_seen=20),
            )
            bridge.cells[("PLACE", "D2")] = "00:00:32"
            update = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )
            bridge.cells[("PLACE", "D2")] = "00:00:20"
            capped_update = provider.place_order(
                replace(
                    _pre_ladder_intent("3/4", ladder_id="ladder-1", trap=1),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=20, milestone_seen=20),
            )

        self.assertEqual([first.status, update.status], ["GRUSS_PRE_LADDER_WRITTEN"] * 2)
        self.assertEqual(capped_update.status, "REJECTED_REAL")
        self.assertEqual(capped_update.reason, "max_orders_reached")
        self.assertEqual(len(bridge.write_calls), 2)

    def test_pre_ladder_real_milestones_are_not_rejected_by_post_countdown_gate(self) -> None:
        for countdown_seconds in (45, 32, 20, 14):
            with self.subTest(countdown_seconds=countdown_seconds):
                bridge = FakeBetRefBridge()
                bridge.cells[("PLACE", "D2")] = f"00:00:{countdown_seconds:02d}"
                with TemporaryDirectory() as tmp, patch.dict("os.environ", PRE_LADDER_REAL_ENV, clear=True):
                    provider = GrussExcelOrderProvider(tmp, bridge=bridge)
                    intent = replace(
                        _pre_ladder_intent("1/4", ladder_id=f"ladder-{countdown_seconds}"),
                        provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                        dry_run=False,
                        stake=2.0,
                        stake_original=5.0,
                        stake_forced=True,
                    )
                    result = provider.place_order(
                        intent,
                        replace(_context(), countdown_seconds=countdown_seconds),
                    )
                    rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

                self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
                self.assertNotIn("countdown_above_3_seconds", result.reason)
                self.assertEqual(rows[0]["countdown_authorization_reason"], "pre_ladder_valid_milestone")

    def test_active_pre_ladder_continues_across_all_configured_milestones(self) -> None:
        bridge = FakeBetRefBridge()
        store = FakeProcessedStore()
        steps = ((45, "1/4", "BACK"), (32, "2/4", "BACKR"), (20, "3/4", "BACKR"), (14, "4/4", "BACKR"))

        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="4",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            all_results = []
            for milestone, ladder_step, _trigger in steps:
                bridge.cells[("PLACE", "D2")] = f"00:00:{milestone:02d}"
                results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                    provider=provider,
                    intents=[_pre_ladder_intent(ladder_step)],
                    context=replace(_context(), countdown_seconds=milestone, milestone_seen=milestone),
                    processed_store=store,
                    key=f"parent:1|milestone={milestone}|phase=PRE",
                    win_market_id="win-1",
                    place_market_id="place-1",
                    force_stake=2.0,
                )
                self.assertFalse(skipped)
                all_results.extend(results)
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual([result.status for result in all_results], ["GRUSS_PRE_LADDER_WRITTEN"] * 4)
        self.assertEqual([result.trigger_value_written for result in all_results], ["BACK", "BACKR", "BACKR", "BACKR"])
        self.assertEqual([result.ladder_step for result in all_results], ["1/4", "2/4", "3/4", "4/4"])
        self.assertEqual(all_results[0].bet_ref_before, "")
        self.assertEqual([result.bet_ref_before for result in all_results[1:]], ["432000000005", "432000000005", "432000000005"])
        self.assertEqual([result.bet_ref_suffix_n_handled for result in all_results], [False, False, True, True])
        self.assertEqual([result.update_allowed for result in all_results], [False, True, True, True])
        self.assertEqual([result.continuing_active_pre_ladder for result in all_results], [False, True, True, True])
        self.assertEqual([row["milestone_seen"] for row in rows], ["45", "32", "20", "14"])
        self.assertEqual([row["next_ladder_step_due"] for row in rows], ["1/4", "2/4", "3/4", "4/4"])
        self.assertEqual([row["continuing_active_pre_ladder"] for row in rows], ["False", "True", "True", "True"])
        self.assertEqual([row["expected_ladder_step"] for row in rows], ["1/4", "2/4", "3/4", "4/4"])
        self.assertEqual([row["configured_ladder_steps"] for row in rows], ["45,32,20,14"] * 4)
        self.assertEqual(rows[-1]["active_ladder_completed"], "True")
        self.assertEqual(rows[-1]["active_ladder_release_reason"], "final_ladder_step_completed")
        self.assertIsNone(provider.active_pre_ladder_id)
        self.assertTrue(all("ladder_step=" in row["processed_key"] for row in rows))
        self.assertEqual(len(bridge.write_calls), 4)

    def test_pre_ladder_replace_without_initial_active_state_attaches_late_bet_ref(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:32"
        bridge.cells[("PLACE", "T5")] = "432000000005"
        bridge.cells[("PLACE", "W5")] = 0.0

        with TemporaryDirectory() as tmp, patch.dict("os.environ", PRE_LADDER_REAL_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    _pre_ladder_intent("2/4"),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=32, milestone_seen=32),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(result.reason, "pre_ladder_replace_written")
        self.assertEqual(result.bet_ref_before, "432000000005")
        self.assertTrue(result.pre_bet_ref_late_detected)
        self.assertEqual(result.pre_bet_ref_late_value, "432000000005")
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.0), ("S5", 2.0), ("Q5", "BACKR")))
        self.assertEqual(rows[0]["reason"], "pre_ladder_replace_written")
        self.assertEqual(rows[0]["bet_ref_lookup_source"], "active_ladder_state_missing_row_t_late_attached")
        self.assertEqual(rows[0]["pre_bet_ref_late_detected"], "True")
        self.assertEqual(rows[0]["pre_bet_ref_late_value"], "432000000005")
        self.assertFalse(result.pre_ladder_initial_order_failed)
        self.assertFalse(result.pre_ladder_disabled_after_initial_failure)
        self.assertFalse(result.no_replace_steps_for_failed_initial)
        self.assertEqual(rows[0]["pre_ladder_initial_order_failed"], "False")
        self.assertEqual(rows[0]["pre_ladder_disabled_after_initial_failure"], "False")
        self.assertEqual(rows[0]["no_replace_steps_for_failed_initial"], "False")

    def test_runner_mapping_scans_past_blank_rows_and_logs_command_cells(self) -> None:
        class BlankMiddleBridge(FakeBetRefBridge):
            def read_range(self, sheet_name, address):
                return [
                    ["1. Runner 1"],
                    [""],
                    ["3. Runner 3"],
                    [None],
                    ["5. Runner 5"],
                    ["6. Runner 6"],
                ]

        bridge = BlankMiddleBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="1",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="1",
        )

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4", ladder_id="ladder-6", trap=6),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(result.excel_row, 10)
        self.assertTrue(result.mapping_found)
        self.assertEqual(result.command_cells, "Q10;R10;S10")
        self.assertEqual(rows[0]["mapping_found"], "True")
        self.assertEqual(rows[0]["mapping_reason"], "mapping_found")
        self.assertEqual(rows[0]["command_cells"], "Q10;R10;S10")

    def test_real_attempt_logs_gruss_matched_metrics_v_w_x_and_requested_values(self) -> None:
        bridge = FakeMatchedMetricsBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"
        env = dict(PRE_LADDER_REAL_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="1")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    replace(_pre_ladder_intent("1/4"), price=2.72),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(result.avg_matched_odds_cell_address, "V5")
        self.assertEqual(result.matched_stake_cell_address, "W5")
        self.assertEqual(result.profit_loss_cell_address, "X5")
        self.assertEqual(result.avg_matched_odds_cell_value, 2.72)
        self.assertEqual(result.matched_stake_cell_value, 1.5)
        self.assertEqual(result.profit_loss_cell_value, -3.0)
        self.assertTrue(result.matched_after_step)
        self.assertEqual(result.matched_after_step_avg_odds, 2.72)
        self.assertEqual(result.matched_after_step_stake, 1.5)
        self.assertEqual(result.requested_price, 2.72)
        self.assertEqual(result.requested_stake, 2.0)
        self.assertEqual(result.ladder_step_index, 1)
        self.assertEqual(result.ladder_step_count, 4)
        self.assertEqual(rows[0]["avg_matched_odds_cell_address"], "V5")
        self.assertEqual(rows[0]["avg_matched_odds_cell_value"], "2.72")
        self.assertEqual(rows[0]["matched_stake_cell_address"], "W5")
        self.assertEqual(rows[0]["matched_stake_cell_value"], "1.5")
        self.assertEqual(rows[0]["profit_loss_cell_address"], "X5")
        self.assertEqual(rows[0]["profit_loss_cell_value"], "-3.0")
        self.assertEqual(rows[0]["matched_after_step"], "True")
        self.assertEqual(rows[0]["requested_price"], "2.72")
        self.assertEqual(rows[0]["requested_stake"], "2.0")

    def test_pre_ladder_excel_runner_range_error_logs_cell_range(self) -> None:
        class BadRunnerRangeBridge(FakeBridge):
            def read_range(self, sheet_name, address):
                if sheet_name == "PLACE" and address == "A5:A84":
                    raise TypeError("This object does not support enumeration")
                return super().read_range(sheet_name, address)

        bridge = BadRunnerRangeBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:45"

        with TemporaryDirectory() as tmp, patch.dict("os.environ", PRE_LADDER_REAL_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    _pre_ladder_intent("1/4"),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake=2.0,
                    stake_original=5.0,
                    stake_forced=True,
                ),
                replace(_context(), countdown_seconds=45, milestone_seen=45),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("excel_mapping_unavailable_after_retries", result.reason)
        self.assertIn("sheet=PLACE range=A5:A84", result.reason)
        self.assertIn("This object does not support enumeration", result.reason)
        self.assertEqual(result.mapping_attempt_count, 4)
        self.assertEqual(result.excel_com_retry_count, 3)
        self.assertEqual(bridge.write_calls, [])

    def test_multi_runner_pre_ladders_each_continue_with_their_own_updates(self) -> None:
        bridge = FakeBetRefBridge()
        store = FakeProcessedStore()
        env = dict(
            PRE_LADDER_REAL_ENV,
            DOGBOT_GRUSS_REAL_MAX_ORDERS="12",
            DOGBOT_PRE_LADDER_REAL_MAX_LADDERS="100",
        )
        steps = ((45, "1/4"), (32, "2/4"), (20, "3/4"), (14, "4/4"))

        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            all_results = []
            for milestone, ladder_step in steps:
                bridge.cells[("PLACE", "D2")] = f"00:00:{milestone:02d}"
                intents = [
                    _pre_ladder_intent(ladder_step, ladder_id=f"ladder-{trap}", trap=trap)
                    for trap in (1, 2, 3)
                ]
                results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                    provider=provider,
                    intents=intents,
                    context=replace(_context(), countdown_seconds=milestone, milestone_seen=milestone),
                    processed_store=store,
                    key=f"parent:1|milestone={milestone}|phase=PRE",
                    win_market_id="win-1",
                    place_market_id="place-1",
                    force_stake=2.0,
                )
                self.assertFalse(skipped)
                all_results.extend(results)

        self.assertEqual([result.status for result in all_results], ["GRUSS_PRE_LADDER_WRITTEN"] * 12)
        self.assertEqual([call[1][2][1] for call in bridge.write_calls[:3]], ["BACK", "BACK", "BACK"])
        self.assertEqual([call[1][-1][1] for call in bridge.write_calls[3:]], ["BACKR"] * 9)
        self.assertEqual(len(bridge.write_calls), 12)
        self.assertFalse(any(call[1][-1][1] == "BACK" for call in bridge.write_calls[3:]))

    def test_post_milestone_releases_active_pre_ladder_if_final_step_was_missed(self) -> None:
        bridge = FakeBetRefBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:01"

        with TemporaryDirectory() as tmp, patch.dict("os.environ", PRE_LADDER_REAL_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            provider.active_pre_ladder_id = "ladder-1"
            provider.active_pre_ladder_course = "parent:1"
            result = provider.place_order(
                replace(
                    _intent(1, stake=2.0),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake_original=2.0,
                    stake_forced=True,
                    execution_phase="POST",
                    selection_id="1",
                ),
                replace(_context(), countdown_seconds=1, milestone_seen=1),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.reason, "excel_trigger_written")
        self.assertIsNone(provider.active_pre_ladder_id)
        self.assertEqual(rows[0]["active_ladder_completed"], "True")
        self.assertEqual(rows[0]["active_ladder_release_reason"], "post_milestone_reached")
        self.assertEqual(rows[0]["active_pre_ladder_id"], "ladder-1")

    def test_milestone_tracker_detects_crossed_pre_ladder_milestones(self) -> None:
        tracker = watch_gruss_real_strategy_test._MilestoneTracker()

        with patch.dict(
            "os.environ",
            dict(PRE_LADDER_REAL_ENV, DOGBOT_STRATEGIES_EXCEL_ENABLED="false"),
            clear=True,
        ):
            self.assertEqual(tracker.due_milestone("race-1", 45, 45), 45)
            self.assertEqual(tracker.due_milestone("race-1", 31, 31), 32)
            self.assertIsNone(tracker.due_milestone("race-1", 31, 31))
            self.assertEqual(tracker.due_milestone("race-1", 19, 19), 20)
            self.assertEqual(tracker.due_milestone("race-1", 11, 11), 14)
            self.assertEqual(tracker.due_milestone("race-1", 1, 1), 1)

    def test_post_classic_still_rejects_countdown_above_three_seconds(self) -> None:
        bridge = FakeBridge()

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    _intent(1, stake=2.0),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake_original=2.0,
                    stake_forced=True,
                    execution_phase="POST",
                    selection_id="1",
                ),
                replace(_context(), countdown_seconds=20),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("countdown_above_3_seconds", result.reason)
        self.assertEqual(bridge.write_calls, [])

    def test_pre_non_ladder_still_rejects_countdown_above_three_seconds(self) -> None:
        bridge = FakeBridge()

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                replace(
                    _intent(1, stake=2.0),
                    provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                    dry_run=False,
                    stake_original=2.0,
                    stake_forced=True,
                    execution_phase="PRE",
                    selection_id="1",
                ),
                replace(_context(), countdown_seconds=20),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("countdown_above_3_seconds", result.reason)
        self.assertEqual(bridge.write_calls, [])


if __name__ == "__main__":
    unittest.main()

