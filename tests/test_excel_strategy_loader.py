from __future__ import annotations

import os
import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dogbot.excel_strategy_loader import (
    CONDITIONS_COLUMNS,
    DEFAULT_STRATEGY_EXCEL_REPORT_PATH,
    GLOBAL_SETTINGS_COLUMNS,
    SHEETS,
    STAKE_PROFILES_COLUMNS,
    STRATEGIES_COLUMNS,
    StrategyExcelConfigError,
    VARIABLES_COLUMNS,
    _read_xlsx,
    _write_xlsx,
    create_strategy_excel_template,
    export_strategy_slots_to_excel,
    load_excel_strategy_slots,
)
from dogbot.strategies import EXECUTION_PHASE_PRE, RunnerCtx, build_registry


def _rows(
    *,
    strategies: list[list[object]] | None = None,
    conditions: list[list[object]] | None = None,
    variables: list[list[object]] | None = None,
    stake_profiles: list[list[object]] | None = None,
    global_settings: list[list[object]] | None = None,
) -> dict[str, list[list[object]]]:
    return {
        "Strategies": [STRATEGIES_COLUMNS, *(strategies or [])],
        "Conditions": [CONDITIONS_COLUMNS, *(conditions or [])],
        "Variables": [
            VARIABLES_COLUMNS,
            *(variables or [
                ["is_row", "ROW flag", "bool", "LIVE", "derived", "TRUE", "TRUE"],
                ["ev_place", "EV place", "number", "LIVE", "ctx", "0.1", "TRUE"],
                ["trap", "Trap", "number", "LIVE", "ctx", "1", "TRUE"],
                ["runner_name", "Runner", "text", "LIVE", "ctx", "Dog", "TRUE"],
                ["bsp_place", "BSP place", "number", "BACKTEST_ONLY", "result", "2.0", "TRUE"],
            ]),
        ],
        "StakeProfiles": [
            STAKE_PROFILES_COLUMNS,
            *(stake_profiles or [["BACK_STANDARD", "VARIABLE", "1", "5", "2", "1", "Back"]]),
        ],
        "GlobalSettings": [
            GLOBAL_SETTINGS_COLUMNS,
            *(global_settings or [["PRE_LADDER_STEPS", "47,30,21,14", "Default PRE ladder"]]),
        ],
        "README": [["README"]],
    }


def _strategy(
    strategy_id: str = "EXCEL_BACK_PLACE_ROW_PRE_001",
    *,
    enabled: str = "TRUE",
    market_type: str = "PLACE",
    side: str = "BACK",
    phase: str = "PRE",
    order_mode: str = "LIMIT",
    price_mode: str = "LIMIT_LTP",
    stake_profile: str = "BACK_STANDARD",
) -> list[object]:
    return [
        enabled,
        strategy_id,
        "Example",
        market_type,
        side,
        phase,
        order_mode,
        price_mode,
        "",
        "",
        "",
        "",
        "",
        "AGGRESSIVE",
        "PLACE_BSP_THEN_LTP",
        "FALSE",
        "",
        "TRUE",
        "EDGE_EXAMPLE",
        "",
        "FALSE",
        "TRUE",
        "EXCEL",
        "",
        "EXCEL",
        "TEST",
        stake_profile,
        "10",
        "",
    ]


def _condition(
    strategy_id: str = "EXCEL_BACK_PLACE_ROW_PRE_001",
    *,
    group: str = "1",
    variable: str = "is_row",
    operator: str = "IS_TRUE",
    value: object = "",
    enabled: str = "TRUE",
) -> list[object]:
    return [enabled, strategy_id, group, variable, operator, value, ""]


def _ctx(**overrides: object) -> RunnerCtx:
    ctx = RunnerCtx(
        market_id="market-1",
        market_type="PLACE",
        selection_id=1,
        course_id="course-1",
        ltp=3.0,
        trap=2,
        region="ROW",
        bb=2.9,
        bl=3.1,
        ev_place=0.1,
        execution_phase=EXECUTION_PHASE_PRE,
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


class ExcelStrategyLoaderTests(unittest.TestCase):
    def test_template_creation_has_required_sheets_columns_and_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            create_strategy_excel_template(path)

            workbook = _read_xlsx(path)
            self.assertEqual(set(SHEETS), set(workbook))
            self.assertEqual(workbook["Strategies"][0], STRATEGIES_COLUMNS)
            self.assertEqual(workbook["Conditions"][0], CONDITIONS_COLUMNS)
            self.assertEqual(workbook["Variables"][0], VARIABLES_COLUMNS)
            self.assertEqual(workbook["StakeProfiles"][0], STAKE_PROFILES_COLUMNS)
            settings = {row[0]: row[1] for row in workbook["GlobalSettings"][1:]}
            self.assertEqual(settings["PRE_LADDER_STEPS"], "47,30,21,14")

    def test_template_script_creates_default_file(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "create_strategy_excel_template.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("strategy Excel template created", result.stdout)
        self.assertTrue((root / "config" / "dogbot_strategies.xlsx").exists())

    def test_export_script_creates_excel_from_current_python_registry(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "export_current_strategies_to_excel.py"
        output = root / "config" / "dogbot_strategies.xlsx"
        report = root / "data" / "strategy_excel_migration_report.csv"
        with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
            python_count = len(build_registry())

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("strategy Excel exported", result.stdout)
        workbook = _read_xlsx(output)
        strategy_rows = workbook["Strategies"][1:]
        self.assertEqual(len(strategy_rows), python_count)
        self.assertGreater(len(strategy_rows), 3)
        strategy_ids = {row[1] for row in strategy_rows}
        self.assertIn("BACK_PLACE_101", strategy_ids)
        self.assertTrue(report.exists())
        with report.open("r", encoding="utf-8", newline="") as handle:
            report_rows = list(csv.DictReader(handle))
        self.assertEqual(len(report_rows), python_count)
        self.assertTrue(all(row["enabled"] == "TRUE" for row in report_rows))
        self.assertTrue(any(row["python_strategy_id"] == "LAY_WIN_401" and row["status"] == "converted" for row in report_rows))

    def test_python_vs_excel_equivalence_script_reports_no_mismatches(self) -> None:
        root = Path(__file__).resolve().parents[1]
        export_script = root / "scripts" / "export_current_strategies_to_excel.py"
        compare_script = root / "scripts" / "compare_python_vs_excel_strategies.py"
        report = root / "data" / "strategy_excel_equivalence_report.csv"

        subprocess.run(
            [sys.executable, str(export_script)],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        result = subprocess.run(
            [sys.executable, str(compare_script)],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("mismatches=0", result.stdout)
        self.assertIn("errors=0", result.stdout)
        self.assertTrue(report.exists())

    def test_export_current_registry_converts_simple_conditions_and_hybrid(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            report = Path(tmp) / "strategy_excel_migration_report.csv"
            with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                slots = build_registry()

            summary = export_strategy_slots_to_excel(slots, path, migration_report_path=report)
            workbook = _read_xlsx(path)
            rows = workbook["Strategies"][1:]

            self.assertEqual(summary["python_strategies_detected"], len(slots))
            self.assertEqual(summary["excel_strategies_exported"], len(slots))
            self.assertEqual(len(rows), len(slots))
            enabled_rows = [row for row in rows if row[0] == "TRUE"]
            disabled_rows = [row for row in rows if row[0] == "FALSE"]
            self.assertEqual(len(enabled_rows), len(slots))
            self.assertEqual(len(disabled_rows), 0)
            load_result = load_excel_strategy_slots(path, report_path=Path(tmp) / "load_report.csv")
            self.assertEqual(load_result.active_count, len(enabled_rows))
            self.assertEqual(load_result.disabled_count, len(disabled_rows))

    def test_exported_python_conditions_include_region_trap_and_price_bounds(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                export_strategy_slots_to_excel(build_registry(), path, migration_report_path=Path(tmp) / "migration.csv")
            workbook = _read_xlsx(path)
            conditions = [dict(zip(workbook["Conditions"][0], row)) for row in workbook["Conditions"][1:]]

            back_place_201 = [row for row in conditions if row["strategy_id"] == "BACK_PLACE_201"]
            self.assertIn(("region", "=", "ROW"), {(row["variable"], row["operator"], str(row["value"])) for row in back_place_201})
            self.assertIn(("place_price_ref", ">=", "1.3"), {(row["variable"], row["operator"], str(row["value"])) for row in back_place_201})
            self.assertIn(("place_price_ref", "<", "3"), {(row["variable"], row["operator"], str(row["value"])) for row in back_place_201})

            trap1 = [row for row in conditions if row["strategy_id"] == "LAY_PLACE_502"]
            self.assertIn(("trap", "=", "1"), {(row["variable"], row["operator"], str(row["value"])) for row in trap1})
            self.assertIn(("ev_place", "<=", "0"), {(row["variable"], row["operator"], str(row["value"])) for row in trap1})

    def test_exported_mom45_place_lay_conditions_include_price_bounds(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                export_strategy_slots_to_excel(build_registry(), path, migration_report_path=Path(tmp) / "migration.csv")
            workbook = _read_xlsx(path)
            conditions = [dict(zip(workbook["Conditions"][0], row)) for row in workbook["Conditions"][1:]]

            expected = {
                "LAY_PLACE_541": ("UK", ">", "7", "<", "-0.15"),
                "LAY_PLACE_542": ("UK", ">", "15", ">", "0.15"),
                "LAY_PLACE_544": ("ROW", ">", "15", "<", "-0.15"),
                "LAY_PLACE_545": ("ROW", ">", "15", ">", "0.15"),
            }
            for strategy_id, (region, price_op, price_value, mom_op, mom_value) in expected.items():
                rows = [row for row in conditions if row["strategy_id"] == strategy_id]
                values = {(row["variable"], row["operator"], str(row["value"])) for row in rows}
                self.assertIn(("region", "=", region), values)
                self.assertIn(("place_price_ref", price_op, price_value), values)
                self.assertIn(("mom_45", mom_op, mom_value), values)

    def test_strategy_metadata_columns_and_hyb_are_exported_and_loaded(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                export_strategy_slots_to_excel(build_registry(), path, migration_report_path=Path(tmp) / "migration.csv")
            workbook = _read_xlsx(path)
            header = workbook["Strategies"][0]
            for column in [
                "edge_env",
                "max_runner_stake_env",
                "price_for_bounds",
                "limit_style",
                "requires_mom45",
                "bet_per_market",
                "strategy_group",
                "strategy_region",
                "strategy_signal",
                "strategy_bucket",
                "price_limit_factor",
                "price_limit_variable",
                "hyb_enabled",
                "hyb_policy",
            ]:
                self.assertIn(column, header)
            rows = [dict(zip(header, row)) for row in workbook["Strategies"][1:]]
            hyb = next(row for row in rows if row["strategy_id"] == "LAY_WIN_401")
            self.assertEqual(hyb["order_mode"], "HYB")
            self.assertEqual(hyb["price_mode"], "HYB_POLICY")
            self.assertEqual(hyb["limit_style"], "AGGRESSIVE")
            self.assertEqual(hyb["price_for_bounds"], "WINBET")
            self.assertEqual(hyb["edge_env"], "EDGE_EV1_WINLAY_ROW")
            self.assertEqual(hyb["max_runner_stake_env"], "MAX_RUNNER_STAKE_EV1_WINLAY_ROW")

            load_result = load_excel_strategy_slots(path, report_path=Path(tmp) / "load_report.csv")
            loaded = next(slot for slot in load_result.slots if slot.tag == "LAY_WIN_401")
            self.assertEqual(loaded.exec_mode.value, "HYB")
            self.assertEqual(loaded.limit_style.value, "AGGRESSIVE")
            self.assertEqual(loaded.price_for_bounds, "WINBET")
            self.assertEqual(loaded.edge_env, "EDGE_EV1_WINLAY_ROW")
            self.assertEqual(loaded.max_runner_stake_env, "MAX_RUNNER_STAKE_EV1_WINLAY_ROW")

    def test_theoretical_limit_factor_is_exported_and_loaded(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                export_strategy_slots_to_excel(build_registry(), path, migration_report_path=Path(tmp) / "migration.csv")
            workbook = _read_xlsx(path)
            rows = [dict(zip(workbook["Strategies"][0], row)) for row in workbook["Strategies"][1:]]
            strategy = next(row for row in rows if row["strategy_id"] == "BACK_PLACE_101")
            self.assertEqual(strategy["price_mode"], "LIMIT_THEO")
            self.assertEqual(strategy["price_limit_variable"], "place_theorique")
            self.assertEqual(float(strategy["price_limit_factor"]), 1.2)

            loaded = next(
                slot
                for slot in load_excel_strategy_slots(path, report_path=Path(tmp) / "load_report.csv").slots
                if slot.tag == "BACK_PLACE_101"
            )
            self.assertEqual(loaded.price_limit_variable, "place_theorique")
            self.assertEqual(float(loaded.price_limit_factor), 1.2)

    def test_export_supports_declarative_excel_conditions_when_available(self) -> None:
        from dogbot.staking import Side
        from dogbot.strategies import ExecMode, Slot

        slot = Slot(
            family="BACK_PLACE",
            slot=999,
            side=Side.BACK,
            condition=lambda ctx: True,
            exec_mode=ExecMode.LIMIT_LTP,
            tag="DECLARATIVE_BACK_PLACE_999",
            market_family="PLACE",
            execution_phase="PRE",
        )
        slot.excel_conditions = [
            {"group": "1", "variable": "is_row", "operator": "IS_TRUE", "value": "", "description": "ROW only"},
            {"group": "1", "variable": "ev_place", "operator": ">", "value": "0", "description": "Positive EV"},
        ]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            export_strategy_slots_to_excel([slot], path, migration_report_path=Path(tmp) / "migration.csv")
            workbook = _read_xlsx(path)
            self.assertEqual(workbook["Strategies"][1][0], "TRUE")
            self.assertEqual(len(workbook["Conditions"][1:]), 2)
            load_result = load_excel_strategy_slots(path, report_path=Path(tmp) / "load.csv")
            self.assertEqual(load_result.active_count, 1)

    def test_loader_loads_active_strategy_and_ignores_disabled(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[
                        _strategy(),
                        _strategy("EXCEL_DISABLED_002", enabled="FALSE"),
                    ],
                    conditions=[_condition()],
                ),
            )

            result = load_excel_strategy_slots(path, report_path=Path(tmp) / "report.csv")

            self.assertEqual(result.active_count, 1)
            self.assertEqual(result.disabled_count, 1)
            self.assertEqual(result.slots[0].tag, "EXCEL_BACK_PLACE_ROW_PRE_001")
            self.assertEqual(result.slots[0].strategy_source, "excel")
            self.assertTrue(result.slots[0].condition(_ctx()))

    def test_loader_writes_load_report(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.xlsx"
            report = Path(tmp) / "strategy_excel_load_report.csv"
            _write_xlsx(path, _rows(strategies=[_strategy()], conditions=[_condition()]))

            load_excel_strategy_slots(path, report_path=report)

            self.assertTrue(report.exists())
            self.assertIn("active_strategies=1", report.read_text(encoding="utf-8"))

    def test_loader_rejects_invalid_strategy_fields(self) -> None:
        cases = [
            ("bad-side", _strategy(side="BID"), [_condition()], "side"),
            ("bad-market", _strategy(market_type="SHOW"), [_condition()], "market_type"),
            ("missing-stake", _strategy(stake_profile="UNKNOWN"), [_condition()], "stake_profile"),
            ("unknown-variable", _strategy(), [_condition(variable="unknown_var")], "unknown variable"),
            ("backtest-variable", _strategy(), [_condition(variable="bsp_place", operator=">", value="1")], "BACKTEST_ONLY"),
            ("unknown-strategy-condition", _strategy(), [_condition(strategy_id="UNKNOWN_STRATEGY")], "unknown strategy_id"),
        ]
        for name, strategy, conditions, expected in cases:
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                path = Path(tmp) / "strategies.xlsx"
                _write_xlsx(path, _rows(strategies=[strategy], conditions=conditions))

                with self.assertRaises(StrategyExcelConfigError) as cm:
                    load_excel_strategy_slots(path, report_path=Path(tmp) / "report.csv")

                self.assertIn(expected, str(cm.exception))

    def test_condition_operators(self) -> None:
        cases = [
            ("=", "runner_name", "=", "dog one", True, {"runner_name": " Dog One "}),
            ("!=", "runner_name", "!=", "other", True, {"runner_name": "Dog One"}),
            (">", "ev_place", ">", "0.05", True, {"ev_place": 0.1}),
            (">=", "ev_place", ">=", "0.1", True, {"ev_place": 0.1}),
            ("<", "ev_place", "<", "0.2", True, {"ev_place": 0.1}),
            ("<=", "ev_place", "<=", "0.1", True, {"ev_place": 0.1}),
            ("IN", "trap", "IN", "1,2,3", True, {"trap": 2}),
            ("NOT_IN", "trap", "NOT_IN", "4,5,6", True, {"trap": 2}),
            ("BETWEEN", "ev_place", "BETWEEN", "0.05,0.2", True, {"ev_place": 0.1}),
            ("IS_TRUE", "is_row", "IS_TRUE", "", True, {"region": "ROW"}),
            ("IS_FALSE", "is_row", "IS_FALSE", "", True, {"region": "UK"}),
        ]
        for name, variable, operator, value, expected, ctx_values in cases:
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                path = Path(tmp) / "strategies.xlsx"
                _write_xlsx(
                    path,
                    _rows(strategies=[_strategy()], conditions=[_condition(variable=variable, operator=operator, value=value)]),
                )
                slot = load_excel_strategy_slots(path, report_path=Path(tmp) / "report.csv").slots[0]
                self.assertEqual(slot.condition(_ctx(**ctx_values)), expected)

    def test_condition_groups_are_and_inside_group_or_between_groups(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[_strategy()],
                    conditions=[
                        _condition(group="1", variable="is_row", operator="IS_TRUE"),
                        _condition(group="1", variable="ev_place", operator=">", value="0.2"),
                        _condition(group="2", variable="trap", operator="=", value="8"),
                    ],
                ),
            )
            slot = load_excel_strategy_slots(path, report_path=Path(tmp) / "report.csv").slots[0]

            self.assertFalse(slot.condition(_ctx(trap=2, ev_place=0.1)))
            self.assertTrue(slot.condition(_ctx(trap=8, ev_place=-0.5)))
            self.assertEqual(getattr(slot.condition, "condition_group_matched"), "2")

    def test_build_registry_falls_back_to_python_when_excel_disabled_or_missing(self) -> None:
        with patch.dict(
            os.environ,
            {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false", "DOGBOT_STRATEGIES_EXCEL_PATH": "missing.xlsx"},
            clear=False,
        ):
            tags = {slot.tag for slot in build_registry()}
        self.assertIn("BACK_PLACE_101", tags)

        with patch.dict(
            os.environ,
            {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "true", "DOGBOT_STRATEGIES_EXCEL_PATH": "missing.xlsx"},
            clear=False,
        ):
            tags = {slot.tag for slot in build_registry()}
        self.assertIn("BACK_PLACE_101", tags)

    def test_build_registry_raises_when_enabled_excel_is_invalid(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.xlsx"
            _write_xlsx(path, _rows(strategies=[_strategy(side="BID")], conditions=[_condition()]))
            with patch.dict(
                os.environ,
                {
                    "DOGBOT_STRATEGIES_EXCEL_ENABLED": "true",
                    "DOGBOT_STRATEGIES_EXCEL_PATH": str(path),
                    "DOGBOT_STRATEGIES_EXCEL_REPORT_PATH": str(Path(tmp) / DEFAULT_STRATEGY_EXCEL_REPORT_PATH.name),
                },
                clear=False,
            ):
                with self.assertRaises(StrategyExcelConfigError):
                    build_registry()


if __name__ == "__main__":
    unittest.main()
