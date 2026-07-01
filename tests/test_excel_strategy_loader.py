from __future__ import annotations

import os
import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import dogbot.excel_strategy_loader as excel_loader
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
    default_strategy_excel_export_path,
    export_strategy_slots_to_excel,
    load_excel_strategy_slots,
    validate_strategy_workbook,
)
from dogbot.strategies import EXECUTION_PHASE_PRE, FUNCTION_REGISTRY, RunnerCtx, build_registry, try_fire_slot


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
                ["market_type", "Market type", "text", "LIVE", "ctx", "PLACE", "TRUE"],
                ["region", "Region", "text", "LIVE", "ctx", "ROW", "TRUE"],
                ["place_price_ref", "Place price", "number", "LIVE", "ctx", "2.5", "TRUE"],
                ["place_theorique", "Place theo", "number", "LIVE", "ctx", "2.0", "TRUE"],
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
            *(global_settings or [["PRE_LADDER_STEPS", "52,38,26,16", "Default PRE ladder"]]),
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
    price_limit_variable: str = "",
    price_limit_factor: str = "",
    stake_profile: str = "BACK_STANDARD",
    function_name: str = "",
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
        price_limit_variable,
        price_limit_factor,
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
        function_name,
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
            self.assertEqual(settings["PRE_LADDER_STEPS"], "52,38,26,16")

    def test_template_script_writes_to_exports_by_default(self) -> None:
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
        output = Path(result.stdout.strip().split("strategy Excel template created:", 1)[1].strip())
        self.assertIn("config", output.parts)
        self.assertIn("exports", output.parts)
        self.assertNotEqual(output.resolve(strict=False), (root / "config" / "dogbot_strategies.xlsx").resolve(strict=False))
        self.assertTrue(output.exists())

    def test_export_script_creates_excel_from_current_python_registry_in_exports_by_default(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "export_current_strategies_to_excel.py"
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
        output = Path(result.stdout.split("strategy Excel exported:", 1)[1].splitlines()[0].strip())
        self.assertIn("config", output.parts)
        self.assertIn("exports", output.parts)
        self.assertNotEqual(output.resolve(strict=False), (root / "config" / "dogbot_strategies.xlsx").resolve(strict=False))
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

        with TemporaryDirectory() as tmp:
            export_path = Path(tmp) / "dogbot_strategies_export.xlsx"
            subprocess.run(
                [sys.executable, str(export_script), "--output", str(export_path)],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            )
            env = dict(os.environ)
            env["DOGBOT_STRATEGIES_EXCEL_PATH"] = str(export_path)
            result = subprocess.run(
                [sys.executable, str(compare_script)],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
                env=env,
            )

        self.assertIn("mismatches=0", result.stdout)
        self.assertIn("errors=0", result.stdout)
        self.assertTrue(report.exists())

    def test_refuses_to_overwrite_active_strategy_config_without_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            active = Path(tmp) / "config" / "dogbot_strategies.xlsx"
            with patch.object(excel_loader, "DEFAULT_STRATEGY_EXCEL_PATH", active):
                with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite config/dogbot_strategies.xlsx without --overwrite-config"):
                    export_strategy_slots_to_excel([], active)

    def test_authorized_active_overwrite_creates_backup(self) -> None:
        with TemporaryDirectory() as tmp:
            active = Path(tmp) / "config" / "dogbot_strategies.xlsx"
            active.parent.mkdir(parents=True)
            create_strategy_excel_template(active, overwrite_config=True, allow_template=True, min_strategies=1)
            with patch.object(excel_loader, "DEFAULT_STRATEGY_EXCEL_PATH", active):
                with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                    slots = build_registry()
                export_strategy_slots_to_excel(slots, active, overwrite_config=True, min_strategies=20)

            backups = list(active.parent.glob("dogbot_strategies_backup_*.xlsx"))
            self.assertEqual(len(backups), 1)
            validation = validate_strategy_workbook(active, min_strategies=20)
            self.assertTrue(validation.ok, validation.issues)

    def test_refuses_to_write_template_to_active_config(self) -> None:
        with TemporaryDirectory() as tmp:
            active = Path(tmp) / "config" / "dogbot_strategies.xlsx"
            with patch.object(excel_loader, "DEFAULT_STRATEGY_EXCEL_PATH", active):
                with self.assertRaisesRegex(RuntimeError, "Template strategy workbook detected"):
                    create_strategy_excel_template(active, overwrite_config=True, min_strategies=20)

    def test_validate_strategy_workbook_accepts_complete_export(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "complete.xlsx"
            with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                export_strategy_slots_to_excel(build_registry(), path, migration_report_path=Path(tmp) / "migration.csv")

            validation = validate_strategy_workbook(path, min_strategies=20)

            self.assertTrue(validation.ok, validation.issues)
            self.assertGreaterEqual(validation.strategies_count, 20)
            self.assertIn("BACK_PLACE_101", validation.strategy_ids)
            self.assertIn("LAY_PLACE_351", validation.strategy_ids)
            self.assertIn("LAY_WIN_401", validation.strategy_ids)
            self.assertIn("BACK_WIN_413", validation.strategy_ids)

    def test_validate_strategy_workbook_accepts_functional_replacements_without_old_required_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "functional_replacements.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[
                        _strategy(
                            "BACK_PLACE_901",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="BACK_PLACE_UK_NON_T1_SMOOTH",
                        ),
                        _strategy(
                            "BACK_PLACE_902",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="BACK_PLACE_ROW_NON_T1_SMOOTH",
                        ),
                        _strategy(
                            "BACK_PLACE_911",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="BACK_PLACE_UK_T1_SMOOTH",
                        ),
                        _strategy(
                            "BACK_PLACE_912",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="BACK_PLACE_ROW_T1_SMOOTH",
                        ),
                    ],
                    conditions=[
                        _condition("BACK_PLACE_901", variable="market_type", operator="=", value="PLACE"),
                        _condition("BACK_PLACE_902", variable="market_type", operator="=", value="PLACE"),
                        _condition("BACK_PLACE_911", variable="market_type", operator="=", value="PLACE"),
                        _condition("BACK_PLACE_912", variable="market_type", operator="=", value="PLACE"),
                    ],
                ),
            )

            validation = validate_strategy_workbook(path, min_strategies=1)

            self.assertTrue(validation.ok, validation.issues)
            self.assertNotIn("BACK_PLACE_101", validation.strategy_ids)
            self.assertFalse(any("missing known active strategy ids" in issue for issue in validation.issues))

    def test_validate_strategy_workbook_rejects_invalid_functional_strategy_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_functional.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[
                        _strategy(
                            "BACK_PLACE_912",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="1.10",
                            function_name="MISSING_FUNCTION",
                        )
                    ],
                    conditions=[
                        _condition("BACK_PLACE_912", variable="market_type", operator="=", value="PLACE"),
                    ],
                ),
            )

            validation = validate_strategy_workbook(path, min_strategies=1)

            self.assertFalse(validation.ok)
            self.assertTrue(any("not found in FUNCTION_REGISTRY" in issue for issue in validation.issues))
            self.assertTrue(any("price_limit_factor=DYNAMIC" in issue for issue in validation.issues))

    def test_validate_strategy_workbook_rejects_excel_only_template(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "template.xlsx"
            create_strategy_excel_template(path)

            validation = validate_strategy_workbook(path, min_strategies=1)

            self.assertFalse(validation.ok)
            self.assertTrue(validation.template_detected)
            self.assertIn("Template strategy workbook detected; refusing to use as active config", validation.issues)

    def test_real_strategy_watcher_refuses_incomplete_active_workbook(self) -> None:
        from scripts.watch_gruss_real_strategy_test import validate_real_strategy_test_environment

        with TemporaryDirectory() as tmp:
            active = Path(tmp) / "config" / "dogbot_strategies.xlsx"
            create_strategy_excel_template(active, overwrite_config=True, allow_template=True, min_strategies=1)
            env = {
                "DOGBOT_DATA_PROVIDER": "gruss_excel",
                "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
                "DOGBOT_GRUSS_REAL_TEST_MODE": "true",
                "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
                "DOGBOT_GRUSS_REAL_MAX_ORDERS": "1",
                "DOGBOT_GRUSS_REAL_MAX_STAKE": "2",
                "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE": "2",
                "DOGBOT_STRATEGIES_EXCEL_ENABLED": "true",
                "DOGBOT_STRATEGIES_EXCEL_PATH": str(active),
            }
            with patch.object(excel_loader, "DEFAULT_STRATEGY_EXCEL_PATH", active):
                with self.assertRaisesRegex(RuntimeError, "Active strategy workbook invalid or incomplete"):
                    validate_real_strategy_test_environment(env)

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

    def test_limit_theo_func_back_row_t1_accepts_and_prices_from_function(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "functional.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[
                        _strategy(
                            "BACK_PLACE_912",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="BACK_PLACE_ROW_T1_SMOOTH",
                        )
                    ],
                    conditions=[
                        _condition("BACK_PLACE_912", variable="market_type", operator="=", value="PLACE"),
                        _condition("BACK_PLACE_912", variable="region", operator="=", value="ROW"),
                        _condition("BACK_PLACE_912", variable="trap", operator="=", value="1"),
                        _condition("BACK_PLACE_912", variable="place_theorique", operator=">", value="1"),
                    ],
                ),
            )
            slot = load_excel_strategy_slots(path, report_path=Path(tmp) / "load.csv").slots[0]
            ctx = _ctx(trap=1, ltp=2.5, place_theo=2.0, ev_place=0.12, region="ROW")

            result = try_fire_slot(None, slot, ctx)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertAlmostEqual(result.price, 2.2)
            self.assertEqual(slot.function_name, "BACK_PLACE_ROW_T1_SMOOTH")
            self.assertEqual(slot.price_limit_factor, "DYNAMIC")
            functional = getattr(slot, "_last_functional_limit_eval")
            self.assertEqual(functional["decision"], "accepted")
            self.assertIsNone(functional["ev_threshold"])
            self.assertAlmostEqual(functional["coeff_limit"], 1.1)
            self.assertAlmostEqual(functional["computed_coefficient"], 1.1)
            self.assertAlmostEqual(functional["computed_limit_price"], 2.2)

    def test_limit_theo_func_back_row_t1_uses_function_as_limit_not_ev_filter(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "functional.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[
                        _strategy(
                            "BACK_PLACE_912",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="BACK_PLACE_ROW_T1_SMOOTH",
                        )
                    ],
                    conditions=[
                        _condition("BACK_PLACE_912", variable="market_type", operator="=", value="PLACE"),
                        _condition("BACK_PLACE_912", variable="region", operator="=", value="ROW"),
                        _condition("BACK_PLACE_912", variable="trap", operator="=", value="1"),
                        _condition("BACK_PLACE_912", variable="place_theorique", operator=">", value="1"),
                    ],
                ),
            )
            slot = load_excel_strategy_slots(path, report_path=Path(tmp) / "load.csv").slots[0]

            result = try_fire_slot(None, slot, _ctx(trap=1, ltp=2.5, place_theo=2.0, ev_place=0.05, region="ROW"))

            self.assertIsNotNone(result)
            assert result is not None
            self.assertAlmostEqual(result.price, 2.2)
            functional = getattr(slot, "_last_functional_limit_eval")
            self.assertEqual(functional["decision"], "accepted")
            self.assertEqual(functional["reason"], "functional_limit_accepted")
            self.assertIsNone(functional["ev_threshold"])
            self.assertAlmostEqual(functional["coeff_limit"], 1.1)

    def test_limit_theo_func_crystal_henry_case_does_not_reject_on_ev_below_target(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "functional_crystal.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[
                        _strategy(
                            "BACK_PLACE_901",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="BACK_PLACE_UK_NON_T1_SMOOTH",
                        )
                    ],
                    conditions=[
                        _condition("BACK_PLACE_901", variable="market_type", operator="=", value="PLACE"),
                        _condition("BACK_PLACE_901", variable="region", operator="=", value="UK"),
                        _condition("BACK_PLACE_901", variable="trap", operator="!=", value="1"),
                        _condition("BACK_PLACE_901", variable="place_theorique", operator=">", value="1"),
                    ],
                ),
            )
            slot = load_excel_strategy_slots(path, report_path=Path(tmp) / "load.csv").slots[0]
            ctx = _ctx(
                trap=2,
                ltp=2.82,
                place_theo=2.44065,
                ev_place=0.15543,
                region="UK",
            )

            result = try_fire_slot(None, slot, ctx)

            self.assertIsNotNone(result)
            assert result is not None
            expected_coefficient = 1.10 + 0.10 * (4.8 - 2.82) / (4.8 - 1.3)
            self.assertAlmostEqual(result.price, 2.44065 * expected_coefficient, places=6)
            functional = getattr(slot, "_last_functional_limit_eval")
            self.assertEqual(functional["decision"], "accepted")
            self.assertEqual(functional["reason"], "functional_limit_accepted")
            self.assertIsNone(functional["ev_threshold"])
            self.assertAlmostEqual(functional["coeff_limit"], expected_coefficient, places=6)

    def test_limit_theo_func_lay_row_t1_accepts_and_rejects_positive_ev(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "functional_lay.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[
                        _strategy(
                            "LAY_PLACE_921",
                            side="LAY",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="LAY_PLACE_ROW_T1_SMOOTH",
                        )
                    ],
                    conditions=[
                        _condition("LAY_PLACE_921", variable="market_type", operator="=", value="PLACE"),
                        _condition("LAY_PLACE_921", variable="region", operator="=", value="ROW"),
                        _condition("LAY_PLACE_921", variable="trap", operator="=", value="1"),
                        _condition("LAY_PLACE_921", variable="place_theorique", operator=">", value="1"),
                    ],
                ),
            )
            slot = load_excel_strategy_slots(path, report_path=Path(tmp) / "load.csv").slots[0]

            accepted = try_fire_slot(None, slot, _ctx(trap=1, ltp=3.0, place_theo=2.0, ev_place=-0.45, region="ROW"))

            self.assertIsNotNone(accepted)
            assert accepted is not None
            self.assertAlmostEqual(accepted.price, 1.1491, places=3)
            functional = getattr(slot, "_last_functional_limit_eval")
            self.assertEqual(functional["decision"], "accepted")
            self.assertIsNone(functional["ev_threshold"])
            self.assertAlmostEqual(functional["coeff_limit"], 0.5748, places=3)

            accepted_even_with_positive_ev = try_fire_slot(
                None,
                slot,
                _ctx(trap=1, ltp=15.0, place_theo=4.0, ev_place=0.078, region="ROW"),
            )

            self.assertIsNotNone(accepted_even_with_positive_ev)
            assert accepted_even_with_positive_ev is not None
            functional = getattr(slot, "_last_functional_limit_eval")
            self.assertEqual(functional["decision"], "accepted")
            self.assertEqual(functional["reason"], "functional_limit_accepted")
            self.assertAlmostEqual(accepted_even_with_positive_ev.price, 4.0 * 0.85)

    def test_post_function_registry_uses_configured_open_thresholds(self) -> None:
        cases = [
            ("BACK_PLACE_POST_UK_NON_T1_OPEN", 2.0, 1.15 + 0.25 * (3.0 - 2.0) / (3.0 - 1.3)),
            ("BACK_PLACE_POST_ROW_NON_T1_OPEN", 2.0, 1.30 + 0.15 * (3.0 - 2.0) / (3.0 - 1.3)),
            ("BACK_PLACE_POST_UK_T1_OPEN", 3.0, 1.05 + 0.20 * (4.8 - 3.0) / (4.8 - 2.0)),
            ("BACK_PLACE_POST_ROW_T1_OPEN", 3.0, 1.30),
        ]
        for function_name, place_odds, expected in cases:
            with self.subTest(function_name=function_name):
                value = FUNCTION_REGISTRY[function_name]["fn"](place_odds)
                self.assertIsNotNone(value)
                self.assertAlmostEqual(value, expected)

    def test_function_coeff_registry_returns_limit_coefficients(self) -> None:
        back_fn = FUNCTION_REGISTRY["BACK_PLACE_UK_NON_T1_SMOOTH"]["fn"]
        lay_fn = FUNCTION_REGISTRY["LAY_PLACE_ROW_T1_SMOOTH"]["fn"]

        self.assertAlmostEqual(back_fn(1.30), 1.20)
        self.assertAlmostEqual(back_fn(2.80), 1.10 + 0.10 * (4.8 - 2.8) / (4.8 - 1.3), places=6)
        self.assertAlmostEqual(back_fn(4.80), 1.10)
        self.assertAlmostEqual(lay_fn(1.05), 0.53)
        self.assertAlmostEqual(lay_fn(7.0), 0.85 - 0.32 * (15.0 - 7.0) / (15.0 - 1.05), places=6)
        self.assertAlmostEqual(lay_fn(15.0), 0.85)

    def test_excel_lay_place_pre_with_true_condition_returns_fire_result(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                export_strategy_slots_to_excel(build_registry(), path, migration_report_path=Path(tmp) / "migration.csv")

            slot = next(
                slot
                for slot in load_excel_strategy_slots(path, report_path=Path(tmp) / "load_report.csv").slots
                if slot.tag == "LAY_PLACE_301"
            )
            ctx = RunnerCtx(
                market_id="place-1",
                market_type="PLACE",
                selection_id=5,
                course_id="course-1",
                ltp=1.94,
                trap=5,
                region="ROW",
                bb=1.94,
                bl=2.18,
                place_theo=1.8195403674848494,
                ev_place=0.06620333061456729,
                execution_phase=EXECUTION_PHASE_PRE,
                milestone=45,
                secs_to_off=45,
            )

            self.assertTrue(slot.condition(ctx))
            self.assertLess(float(slot.sp_limit_fn(ctx)), 1.0)
            result = try_fire_slot(None, slot, ctx)

            self.assertIsNotNone(result)
            self.assertEqual(result.price, 1.01)
            self.assertLess(float(result.sp_limit), 1.0)
            self.assertGreater(result.size, 0.0)

    def test_excel_back_and_lay_pre_candidates_both_reach_conflict_resolution(self) -> None:
        from dogbot.executor import _StrategyOrderCandidate, _resolve_back_lay_same_phase_candidates

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dogbot_strategies.xlsx"
            with patch.dict(os.environ, {"DOGBOT_STRATEGIES_EXCEL_ENABLED": "false"}, clear=False):
                export_strategy_slots_to_excel(build_registry(), path, migration_report_path=Path(tmp) / "migration.csv")
            slots = load_excel_strategy_slots(path, report_path=Path(tmp) / "load_report.csv").slots
            back_slot = next(slot for slot in slots if slot.tag == "BACK_PLACE_201")
            lay_slot = next(slot for slot in slots if slot.tag == "LAY_PLACE_301")

            ctx = RunnerCtx(
                market_id="place-1",
                market_type="PLACE",
                selection_id=5,
                course_id="course-1",
                ltp=1.94,
                trap=5,
                region="ROW",
                bb=1.94,
                bl=2.18,
                place_theo=1.8195403674848494,
                ev_place=0.06620333061456729,
                execution_phase=EXECUTION_PHASE_PRE,
                milestone=45,
                secs_to_off=45,
            )
            back_result = try_fire_slot(None, back_slot, ctx)
            lay_result = try_fire_slot(None, lay_slot, ctx)

            self.assertTrue(back_slot.condition(ctx))
            self.assertTrue(lay_slot.condition(ctx))
            self.assertIsNotNone(back_result)
            self.assertIsNotNone(lay_result)

            def candidate(slot, result):
                return _StrategyOrderCandidate(
                    slot=slot,
                    market_id=ctx.market_id,
                    market_type=ctx.market_type,
                    selection_id=ctx.selection_id,
                    course_id=ctx.course_id,
                    side=slot.side.value,
                    price=float(result.price),
                    size=float(result.size),
                    liability=round(float(result.liability or 0.0), 2),
                    reason=result.reason,
                    exec_mode=result.exec_mode,
                    sp_limit=result.sp_limit,
                    execution_phase=EXECUTION_PHASE_PRE,
                    triggered_systems=[str(slot.tag)],
                    triggered_prices=[float(result.price)],
                    bet_per_market_key=(slot.family, slot.slot, ctx.market_id),
                    best_unmatched_back_offer=ctx.bb,
                    best_unmatched_lay_offer=ctx.bl,
                )

            raw_candidates = [candidate(back_slot, back_result), candidate(lay_slot, lay_result)]
            selected, rejected = _resolve_back_lay_same_phase_candidates(raw_candidates)

            self.assertEqual({candidate.side for candidate in raw_candidates}, {"BACK", "LAY"})
            self.assertEqual(len(selected), 1)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(selected[0].side, "BACK")
            self.assertEqual(rejected[0].reason, "conflicting_back_lay_lost_priority")
            self.assertEqual(selected[0].pre_conflict_reason, "pre_conflict_back_nearer")

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

    def test_loader_report_includes_loaded_workbook_identity_and_functional_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.xlsx"
            report = Path(tmp) / "strategy_excel_load_report.csv"
            _write_xlsx(
                path,
                _rows(
                    strategies=[
                        _strategy(
                            "BACK_PLACE_901",
                            price_mode="LIMIT_THEO_FUNC",
                            price_limit_variable="place_theorique",
                            price_limit_factor="DYNAMIC",
                            function_name="BACK_PLACE_UK_T1_SMOOTH",
                        )
                    ],
                    conditions=[_condition("BACK_PLACE_901", variable="market_type", operator="=", value="PLACE")],
                ),
            )

            result = load_excel_strategy_slots(path, report_path=report)

            self.assertEqual(result.first_strategy_ids, ["BACK_PLACE_901"])
            self.assertEqual(result.limit_theo_func_count, 1)
            self.assertEqual(result.function_name_count, 1)
            self.assertEqual(result.functional_strategy_ids, ["BACK_PLACE_901"])
            with report.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["absolute_excel_path"], str(path.resolve(strict=False)))
            self.assertEqual(rows[0]["strategies_count"], "1")
            self.assertEqual(rows[0]["active_strategies"], "1")
            self.assertEqual(rows[0]["first_strategy_ids"], "BACK_PLACE_901")
            self.assertEqual(rows[0]["contains_LIMIT_THEO_FUNC_count"], "1")
            self.assertEqual(rows[0]["contains_function_name_count"], "1")
            self.assertEqual(rows[0]["functional_strategy_ids"], "BACK_PLACE_901")

    def test_loader_rejects_strategy_editor_and_strategies_id_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.xlsx"
            rows = _rows(strategies=[_strategy("BACK_PLACE_901")], conditions=[_condition("BACK_PLACE_901")])
            rows["Strategy_Editor"] = [
                ["strategy_id", "enabled"],
                ["BACK_PLACE_902", "TRUE"],
            ]
            _write_xlsx(path, rows)

            with self.assertRaisesRegex(StrategyExcelConfigError, "Strategy_Editor and Strategies strategy_id sets differ"):
                load_excel_strategy_slots(path, report_path=Path(tmp) / "report.csv")

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
            ("==", "runner_name", "==", "dog one", True, {"runner_name": " Dog One "}),
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

    def test_trap_not_equal_conditions_exclude_only_configured_traps(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.xlsx"
            _write_xlsx(
                path,
                _rows(
                    strategies=[_strategy("LAY_PLACE_302", side="LAY")],
                    conditions=[
                        _condition("LAY_PLACE_302", variable="trap", operator="!=", value="1"),
                        _condition("LAY_PLACE_302", variable="trap", operator="!=", value="8"),
                    ],
                ),
            )
            slot = load_excel_strategy_slots(path, report_path=Path(tmp) / "report.csv").slots[0]

            self.assertFalse(slot.condition(_ctx(trap=1)))
            self.assertFalse(slot.condition(_ctx(trap=8)))
            for trap in (2, 3, 4, 5, 6, 7):
                with self.subTest(trap=trap):
                    self.assertTrue(slot.condition(_ctx(trap=trap)))

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
