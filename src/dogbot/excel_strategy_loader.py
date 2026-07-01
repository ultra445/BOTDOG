from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
import csv
import math
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET

from .staking import Side
from .strategies import (
    EXECUTION_PHASE_POST,
    EXECUTION_PHASE_PRE,
    FUNCTION_REGISTRY,
    ExecMode,
    LimitStyle,
    RunnerCtx,
    Slot,
)


DEFAULT_STRATEGY_EXCEL_PATH = Path("config/dogbot_strategies.xlsx")
DEFAULT_STRATEGY_EXCEL_EXPORT_DIR = Path("config/exports")
DEFAULT_STRATEGY_EXCEL_REPORT_PATH = Path("data/strategy_excel_load_report.csv")
DEFAULT_STRATEGY_EXCEL_MIGRATION_REPORT_PATH = Path("data/strategy_excel_migration_report.csv")

SHEETS = ["Strategies", "Conditions", "Variables", "StakeProfiles", "GlobalSettings", "README"]
REQUIRED_ACTIVE_SHEETS = ["Strategies", "Conditions", "Variables", "StakeProfiles", "GlobalSettings"]

STRATEGIES_COLUMNS = [
    "enabled",
    "strategy_id",
    "name",
    "market_type",
    "side",
    "phase",
    "order_mode",
    "price_mode",
    "sp_limit_price",
    "sp_limit_variable",
    "sp_limit_multiplier",
    "price_limit_variable",
    "price_limit_factor",
    "limit_style",
    "price_for_bounds",
    "hyb_enabled",
    "hyb_policy",
    "hyb_fallback_to_sp_moc",
    "edge_env",
    "max_runner_stake_env",
    "requires_mom45",
    "bet_per_market",
    "strategy_group",
    "strategy_region",
    "strategy_signal",
    "strategy_bucket",
    "stake_profile",
    "priority",
    "description",
    "function_name",
]
OPTIONAL_STRATEGIES_COLUMNS = {"function_name"}
CONDITIONS_COLUMNS = ["enabled", "strategy_id", "group", "variable", "operator", "value", "description"]
VARIABLES_COLUMNS = ["variable", "description", "type", "availability", "source", "example", "enabled"]
STAKE_PROFILES_COLUMNS = ["stake_profile", "mode", "min_stake", "max_stake", "base_stake", "alpha", "description"]
GLOBAL_SETTINGS_COLUMNS = ["key", "value", "description"]
TEMPLATE_STRATEGY_IDS = {
    "EXCEL_BACK_PLACE_ROW_PRE_001",
    "EXCEL_LAY_PLACE_ROW_PRE_001",
    "EXCEL_LAY_PLACE_UK_PRE_001",
}

REPORT_COLUMNS = [
    "timestamp",
    "status",
    "sheet",
    "row",
    "strategy_id",
    "field",
    "message",
    "severity",
    "excel_path",
    "absolute_excel_path",
    "file_modified_time",
    "file_size",
    "strategies_count",
    "active_strategies",
    "disabled_strategies",
    "first_strategy_ids",
    "contains_LIMIT_THEO_FUNC_count",
    "contains_function_name_count",
    "functional_strategy_ids",
]
MIGRATION_REPORT_COLUMNS = [
    "timestamp",
    "python_strategy_id",
    "excel_strategy_id",
    "name",
    "status",
    "enabled",
    "market_type",
    "side",
    "phase",
    "migrated_conditions_count",
    "warning",
    "message",
]

ALLOWED_CONDITION_OPERATORS = {
    "=",
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "IN",
    "NOT_IN",
    "BETWEEN",
    "IS_TRUE",
    "IS_FALSE",
    "IS_EMPTY",
    "IS_NOT_EMPTY",
}

LIVE_VARIABLES: list[tuple[str, str, str, str, str, str, str]] = [
    ("region", "UK / ROW", "text", "LIVE", "RunnerCtx.region", "UK", "TRUE"),
    ("country_code", "Country code when available", "text", "LIVE", "derived", "UK", "TRUE"),
    ("market_type", "WIN / PLACE", "text", "LIVE", "RunnerCtx.market_type", "PLACE", "TRUE"),
    ("trap", "Trap number", "number", "LIVE", "RunnerCtx.trap", "1", "TRUE"),
    ("runner_name", "Runner name if supplied by caller", "text", "LIVE", "derived", "1. Example", "TRUE"),
    ("selection_id", "Betfair selection id", "number", "LIVE", "RunnerCtx.selection_id", "12345", "TRUE"),
    ("market_id", "Betfair market id", "text", "LIVE", "RunnerCtx.market_id", "1.234", "TRUE"),
    ("course_id", "Course/race key", "text", "LIVE", "RunnerCtx.course_id", "Romford-...", "TRUE"),
    ("distance_m", "Race distance in metres if available", "number", "LIVE", "derived", "480", "TRUE"),
    ("runners_count", "Runner count if available", "number", "LIVE", "derived", "6", "TRUE"),
    ("countdown", "Seconds to off", "number", "LIVE", "RunnerCtx.secs_to_off", "45", "TRUE"),
    ("ltp", "Last traded price", "number", "LIVE", "RunnerCtx.ltp", "3.2", "TRUE"),
    ("bb", "Best back", "number", "LIVE", "RunnerCtx.bb", "3.15", "TRUE"),
    ("bl", "Best lay", "number", "LIVE", "RunnerCtx.bl", "3.2", "TRUE"),
    ("best_back", "Best back alias", "number", "LIVE", "RunnerCtx.bb", "3.15", "TRUE"),
    ("best_lay", "Best lay alias", "number", "LIVE", "RunnerCtx.bl", "3.2", "TRUE"),
    ("win_price_ref", "WIN price reference equivalent to _pick_bounds_price(ctx, WINBET)", "number", "LIVE", "RunnerCtx.winbet/base_win/ltp", "4.0", "TRUE"),
    ("place_price_ref", "PLACE price reference equivalent to _pick_bounds_price(ctx, PLACE_BSP_THEN_LTP)", "number", "LIVE", "RunnerCtx.bsp_place/ltp", "2.0", "TRUE"),
    ("cote_win", "WIN price hierarchy", "number", "LIVE", "RunnerCtx.winbet", "4.0", "TRUE"),
    ("cote_place", "PLACE price", "number", "LIVE", "RunnerCtx.ltp", "2.0", "TRUE"),
    ("ltp_win", "WIN LTP/base price", "number", "LIVE", "RunnerCtx.base_win", "4.0", "TRUE"),
    ("ltp_place", "PLACE LTP", "number", "LIVE", "RunnerCtx.ltp", "2.0", "TRUE"),
    ("winbet", "WINBET hierarchy", "number", "LIVE", "RunnerCtx.winbet", "4.0", "TRUE"),
    ("place_theorique", "Theoretical place odds", "number", "LIVE", "RunnerCtx.place_theo", "2.2", "TRUE"),
    ("placetheorique_p", "Theoretical place odds alias", "number", "LIVE", "RunnerCtx.place_theo", "2.2", "TRUE"),
    ("ev_place", "PLACE EV", "number", "LIVE", "RunnerCtx.ev_place", "0.08", "TRUE"),
    ("ratio_place", "PLACE ratio if available", "number", "LIVE", "derived", "1.05", "TRUE"),
    ("partenjeuxplace_p", "PLACE stake share if available", "number", "LIVE", "derived", "0.12", "TRUE"),
    ("mom_45", "MOM45", "number", "LIVE", "RunnerCtx.mom45", "0.05", "TRUE"),
    ("mom_30", "MOM30/d30", "number", "LIVE", "RunnerCtx.d30", "0.03", "TRUE"),
    ("mom_20", "MOM20 if available", "number", "LIVE", "derived", "0.02", "TRUE"),
    ("mom_15", "MOM15 if available", "number", "LIVE", "derived", "0.01", "TRUE"),
    ("milestone", "Current strategy milestone", "number", "LIVE", "RunnerCtx.milestone", "2", "TRUE"),
    ("is_uk", "Region is UK", "bool", "LIVE", "derived", "TRUE", "TRUE"),
    ("is_row", "Region is ROW", "bool", "LIVE", "derived", "FALSE", "TRUE"),
    ("is_win", "Market is WIN", "bool", "LIVE", "derived", "FALSE", "TRUE"),
    ("is_place", "Market is PLACE", "bool", "LIVE", "derived", "TRUE", "TRUE"),
    ("is_pre", "Execution phase is PRE", "bool", "LIVE", "derived", "TRUE", "TRUE"),
    ("is_post", "Execution phase is POST", "bool", "LIVE", "derived", "FALSE", "TRUE"),
    ("is_trap_1", "Trap equals 1", "bool", "LIVE", "derived", "TRUE", "TRUE"),
    ("is_trap_8", "Trap equals 8", "bool", "LIVE", "derived", "FALSE", "TRUE"),
    ("bsp_win", "Final BSP WIN", "number", "BACKTEST_ONLY", "results", "4.2", "TRUE"),
    ("bsp_place", "Final BSP PLACE", "number", "BACKTEST_ONLY", "results", "2.1", "TRUE"),
    ("result", "Result", "text", "BACKTEST_ONLY", "results", "WON", "TRUE"),
    ("profit_loss", "P/L", "number", "BACKTEST_ONLY", "results", "1.5", "TRUE"),
    ("winner", "Winner flag", "bool", "BACKTEST_ONLY", "results", "FALSE", "TRUE"),
    ("placed", "Placed flag", "bool", "BACKTEST_ONLY", "results", "TRUE", "TRUE"),
]


@dataclass
class StrategyExcelIssue:
    status: str
    sheet: str
    row: int | str
    strategy_id: str
    field: str
    message: str
    severity: str = "ERROR"


@dataclass
class StrategyExcelLoadResult:
    slots: list[Slot]
    issues: list[StrategyExcelIssue]
    strategies_read: int
    active_count: int
    disabled_count: int
    conditions_read: int
    global_settings: dict[str, str]
    excel_path: str = ""
    absolute_excel_path: str = ""
    file_modified_time: str = ""
    file_size: int = 0
    first_strategy_ids: list[str] | None = None
    limit_theo_func_count: int = 0
    function_name_count: int = 0
    functional_strategy_ids: list[str] | None = None

    @property
    def errors(self) -> list[StrategyExcelIssue]:
        return [issue for issue in self.issues if issue.severity.upper() == "ERROR"]


@dataclass
class StrategyWorkbookValidationResult:
    path: Path
    ok: bool
    issues: list[str]
    strategies_count: int
    enabled_count: int
    back_count: int
    lay_count: int
    pre_count: int
    post_count: int
    win_count: int
    place_count: int
    strategy_ids: list[str]
    template_detected: bool

    @property
    def first_strategy_ids(self) -> list[str]:
        return self.strategy_ids[:10]


class StrategyExcelConfigError(RuntimeError):
    def __init__(self, issues: Iterable[StrategyExcelIssue]):
        self.issues = list(issues)
        details = "; ".join(
            f"{issue.sheet}!row={issue.row} strategy={issue.strategy_id or '-'} "
            f"{issue.field}: {issue.message}"
            for issue in self.issues[:10]
            if issue.severity.upper() == "ERROR"
        )
        super().__init__(details or "invalid strategy Excel configuration")


def timestamp_for_strategy_export() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_strategy_excel_export_path(prefix: str = "dogbot_strategies_export") -> Path:
    return DEFAULT_STRATEGY_EXCEL_EXPORT_DIR / f"{prefix}_{timestamp_for_strategy_export()}.xlsx"


def is_active_strategy_config_path(path: Path | str) -> bool:
    candidate = Path(path)
    active = DEFAULT_STRATEGY_EXCEL_PATH
    try:
        return candidate.resolve(strict=False) == active.resolve(strict=False)
    except OSError:
        return candidate.as_posix().lower() == active.as_posix().lower()


def strategy_config_backup_path(path: Path | str = DEFAULT_STRATEGY_EXCEL_PATH) -> Path:
    target = Path(path)
    return target.with_name(f"{target.stem}_backup_{timestamp_for_strategy_export()}{target.suffix}")


def backup_strategy_config(path: Path | str = DEFAULT_STRATEGY_EXCEL_PATH) -> Path | None:
    target = Path(path)
    if not target.exists():
        return None
    backup = strategy_config_backup_path(target)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, backup)
    return backup


def _workbook_file_metadata(path: Path | str) -> dict[str, Any]:
    workbook_path = Path(path)
    try:
        resolved = workbook_path.resolve(strict=False)
    except OSError:
        resolved = workbook_path
    try:
        stat = workbook_path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")
        size = int(stat.st_size)
    except OSError:
        modified = ""
        size = 0
    return {
        "excel_path": str(workbook_path),
        "absolute_excel_path": str(resolved),
        "file_modified_time": modified,
        "file_size": size,
    }


def _strategy_workbook_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [
        _norm_text(row.get("strategy_id"))
        for row in rows
        if _norm_text(row.get("strategy_id"))
    ]
    functional_ids = [
        _norm_text(row.get("strategy_id"))
        for row in rows
        if _norm_text(row.get("strategy_id"))
        and _norm_text(row.get("price_mode")).upper() == "LIMIT_THEO_FUNC"
    ]
    return {
        "first_strategy_ids": ids[:10],
        "limit_theo_func_count": len(functional_ids),
        "function_name_count": sum(1 for row in rows if _norm_text(row.get("function_name"))),
        "functional_strategy_ids": functional_ids,
    }


def validate_strategy_workbook(
    path: Path | str,
    *,
    min_strategies: int = 20,
    allow_template: bool = False,
) -> StrategyWorkbookValidationResult:
    workbook_path = Path(path)
    issues: list[str] = []
    strategies_count = enabled_count = back_count = lay_count = pre_count = post_count = win_count = place_count = 0
    strategy_ids: list[str] = []
    template_detected = False
    if not workbook_path.exists():
        return StrategyWorkbookValidationResult(
            path=workbook_path,
            ok=False,
            issues=[f"file_not_found={workbook_path}"],
            strategies_count=0,
            enabled_count=0,
            back_count=0,
            lay_count=0,
            pre_count=0,
            post_count=0,
            win_count=0,
            place_count=0,
            strategy_ids=[],
            template_detected=False,
        )
    try:
        sheets = _read_xlsx(workbook_path)
    except Exception as exc:
        return StrategyWorkbookValidationResult(
            path=workbook_path,
            ok=False,
            issues=[f"read_failed={exc}"],
            strategies_count=0,
            enabled_count=0,
            back_count=0,
            lay_count=0,
            pre_count=0,
            post_count=0,
            win_count=0,
            place_count=0,
            strategy_ids=[],
            template_detected=False,
        )

    for sheet in REQUIRED_ACTIVE_SHEETS:
        if sheet not in sheets:
            issues.append(f"missing required sheet: {sheet}")
    variables_sheet = sheets.get("Variables") or []
    variable_names = {
        _norm_text(row.get("variable"))
        for row in (_rows_as_dicts(variables_sheet) if variables_sheet else [])
        if _norm_text(row.get("variable"))
    }
    strategies_sheet = sheets.get("Strategies") or []
    if strategies_sheet:
        header = [str(value or "").strip() for value in strategies_sheet[0]]
        missing_columns = [column for column in ["enabled", "strategy_id", "market_type", "side", "phase", "order_mode", "price_mode", "stake_profile"] if column not in header]
        if missing_columns:
            issues.append(f"missing Strategies columns: {','.join(missing_columns)}")
        rows = _rows_as_dicts(strategies_sheet)
        seen_strategy_ids: set[str] = set()
        for row_index, row in enumerate(rows, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            sid = _norm_text(row.get("strategy_id"))
            if not sid:
                issues.append(f"Strategies row {row_index}: missing strategy_id")
                continue
            if sid in seen_strategy_ids:
                issues.append(f"Strategies row {row_index}: duplicate strategy_id {sid}")
            seen_strategy_ids.add(sid)
            strategy_ids.append(sid)
            strategies_count += 1
            if _parse_bool(row.get("enabled"), default=False):
                enabled_count += 1
            side = _norm_text(row.get("side")).upper()
            phase = _norm_text(row.get("phase")).upper()
            market_type = _norm_text(row.get("market_type")).upper()
            order_mode = _norm_text(row.get("order_mode")).upper()
            price_mode = _norm_text(row.get("price_mode")).upper()
            if market_type not in {"WIN", "PLACE"}:
                issues.append(f"Strategies row {row_index} {sid}: market_type must be WIN or PLACE")
            if side not in {"BACK", "LAY"}:
                issues.append(f"Strategies row {row_index} {sid}: side must be BACK or LAY")
            if phase not in {EXECUTION_PHASE_PRE, EXECUTION_PHASE_POST}:
                issues.append(f"Strategies row {row_index} {sid}: phase must be PRE or POST")
            if order_mode == "LIMIT" and price_mode not in {"LIMIT_LTP", "LIMIT_THEO", "LIMIT_THEO_FUNC"}:
                issues.append(f"Strategies row {row_index} {sid}: LIMIT requires LIMIT_LTP, LIMIT_THEO or LIMIT_THEO_FUNC")
            if price_mode == "LIMIT_THEO_FUNC":
                function_name = _norm_text(row.get("function_name")).upper()
                price_limit_factor = _norm_text(row.get("price_limit_factor")).upper()
                if not function_name:
                    issues.append(f"Strategies row {row_index} {sid}: LIMIT_THEO_FUNC requires function_name")
                elif function_name not in FUNCTION_REGISTRY:
                    issues.append(f"Strategies row {row_index} {sid}: function_name {function_name!r} not found in FUNCTION_REGISTRY")
                if price_limit_factor != "DYNAMIC":
                    issues.append(f"Strategies row {row_index} {sid}: LIMIT_THEO_FUNC requires price_limit_factor=DYNAMIC")
            if side == "BACK":
                back_count += 1
            if side == "LAY":
                lay_count += 1
            if phase == EXECUTION_PHASE_PRE:
                pre_count += 1
            if phase == EXECUTION_PHASE_POST:
                post_count += 1
            if market_type == "WIN":
                win_count += 1
            if market_type == "PLACE":
                place_count += 1
    else:
        issues.append("missing Strategies sheet data")
    conditions_sheet = sheets.get("Conditions") or []
    if conditions_sheet:
        condition_rows = _rows_as_dicts(conditions_sheet)
        id_set_for_conditions = set(strategy_ids)
        for row_index, row in enumerate(condition_rows, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            sid = _norm_text(row.get("strategy_id"))
            variable = _norm_text(row.get("variable"))
            operator = _norm_text(row.get("operator")).upper()
            if sid and sid not in id_set_for_conditions:
                issues.append(f"Conditions row {row_index}: orphan condition for {sid}")
            if not operator:
                issues.append(f"Conditions row {row_index} {sid}: missing operator")
            elif operator not in ALLOWED_CONDITION_OPERATORS:
                issues.append(f"Conditions row {row_index} {sid}: unsupported operator {operator!r}")
            if variable and variable_names and variable not in variable_names:
                issues.append(f"Conditions row {row_index} {sid}: unknown variable {variable!r}")

    if strategies_count < min_strategies:
        issues.append(f"strategies_count={strategies_count} below min_strategies={min_strategies}")
    id_set = set(strategy_ids)
    template_detected = bool(strategy_ids) and (
        id_set <= TEMPLATE_STRATEGY_IDS or all(strategy_id.startswith("EXCEL_") for strategy_id in strategy_ids)
    )
    if template_detected and not allow_template:
        issues.append("Template strategy workbook detected; refusing to use as active config")

    return StrategyWorkbookValidationResult(
        path=workbook_path,
        ok=not issues,
        issues=issues,
        strategies_count=strategies_count,
        enabled_count=enabled_count,
        back_count=back_count,
        lay_count=lay_count,
        pre_count=pre_count,
        post_count=post_count,
        win_count=win_count,
        place_count=place_count,
        strategy_ids=strategy_ids,
        template_detected=template_detected,
    )


def _write_strategy_workbook_guarded(
    target: Path,
    rows: dict[str, list[list[Any]]],
    *,
    overwrite_config: bool = False,
    allow_template: bool = False,
    min_strategies: int = 20,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_active_strategy_config_path(target):
        if not overwrite_config:
            raise RuntimeError("Refusing to overwrite config/dogbot_strategies.xlsx without --overwrite-config")
        pending = target.with_name(f".{target.stem}_pending_{timestamp_for_strategy_export()}{target.suffix}")
        _write_xlsx(pending, rows)
        validation = validate_strategy_workbook(pending, min_strategies=min_strategies, allow_template=allow_template)
        if not validation.ok:
            try:
                pending.unlink()
            except OSError:
                pass
            raise RuntimeError("Refusing to write invalid active strategy workbook: " + "; ".join(validation.issues))
        backup_strategy_config(target)
        pending.replace(target)
        return target
    _write_xlsx(target, rows)
    return target


def create_strategy_excel_template(
    path: Path | str | None = None,
    *,
    overwrite_config: bool = False,
    allow_template: bool = False,
    min_strategies: int = 1,
) -> Path:
    if path is None:
        target = default_strategy_excel_export_path("dogbot_strategies_template")
    else:
        target = Path(path)
    rows = {
        "Strategies": [
            STRATEGIES_COLUMNS,
            ["TRUE", "EXCEL_BACK_PLACE_ROW_PRE_001", "Example PLACE BACK ROW PRE LIMIT", "PLACE", "BACK", "PRE", "LIMIT", "LIMIT_LTP", "", "", "", "", "", "AGGRESSIVE", "PLACE_BSP_THEN_LTP", "FALSE", "", "TRUE", "EDGE_EXAMPLE_BACK", "", "FALSE", "FALSE", "EXCEL", "ROW", "EV_PLACE", "PLACE_1.3_3.0", "BACK_STANDARD", "10", "Example: ROW PLACE BACK when EV is positive"],
            ["TRUE", "EXCEL_LAY_PLACE_ROW_PRE_001", "Example PLACE LAY ROW PRE LIMIT", "PLACE", "LAY", "PRE", "LIMIT", "LIMIT_LTP", "", "", "", "", "", "AGGRESSIVE", "PLACE_BSP_THEN_LTP", "FALSE", "", "TRUE", "EDGE_EXAMPLE_LAY", "", "FALSE", "FALSE", "EXCEL", "ROW", "EV_PLACE", "PLACE_1.05_3.0", "LAY_STANDARD", "20", "Example: ROW PLACE LAY at low EV"],
            ["FALSE", "EXCEL_LAY_PLACE_UK_PRE_001", "Example PLACE LAY UK PRE LIMIT", "PLACE", "LAY", "PRE", "LIMIT", "LIMIT_THEO", "", "", "", "place_theorique", "0.80", "AGGRESSIVE", "PLACE_BSP_THEN_LTP", "FALSE", "", "TRUE", "EDGE_EXAMPLE_LAY_UK", "", "FALSE", "FALSE", "EXCEL", "UK", "EV_PLACE", "PLACE_15_PLUS", "LAY_STANDARD", "30", "Disabled example using theoretical price"],
        ],
        "Conditions": [
            CONDITIONS_COLUMNS,
            ["TRUE", "EXCEL_BACK_PLACE_ROW_PRE_001", "1", "is_row", "IS_TRUE", "", "Only ROW"],
            ["TRUE", "EXCEL_BACK_PLACE_ROW_PRE_001", "1", "ev_place", ">", "0", "Positive EV"],
            ["TRUE", "EXCEL_LAY_PLACE_ROW_PRE_001", "1", "is_row", "IS_TRUE", "", "Only ROW"],
            ["TRUE", "EXCEL_LAY_PLACE_ROW_PRE_001", "1", "ev_place", "<", "0", "Negative EV"],
            ["TRUE", "EXCEL_LAY_PLACE_UK_PRE_001", "1", "is_uk", "IS_TRUE", "", "Only UK"],
            ["TRUE", "EXCEL_LAY_PLACE_UK_PRE_001", "1", "trap", "IN", "1,2,3,4,5,6", "Normal traps"],
        ],
        "Variables": [VARIABLES_COLUMNS, *[list(row) for row in LIVE_VARIABLES]],
        "StakeProfiles": [
            STAKE_PROFILES_COLUMNS,
            ["FIXED_1", "FIXED", "1", "1", "1", "", "Fixed stake 1"],
            ["FIXED_2", "FIXED", "1", "2", "2", "", "Fixed stake 2"],
            ["BACK_STANDARD", "VARIABLE", "1", "5", "2", "1", "Standard BACK variable stake"],
            ["LAY_STANDARD", "VARIABLE", "1", "5", "2", "1", "Standard LAY variable stake"],
        ],
        "GlobalSettings": [
            GLOBAL_SETTINGS_COLUMNS,
            ["PRE_LADDER_STEPS", "52,38,26,16", "Default PRE ladder steps"],
            ["POST_SEND_SECONDS", "12", "Informational V1 setting"],
            ["MIN_STAKE", "1", "Minimum real stake"],
            ["MAX_STAKE", "5", "Maximum real stake"],
            ["PRE_POST_INDEPENDENT", "TRUE", "PRE and POST can both run"],
            ["PRE_CANCEL_BEFORE_POST", "FALSE", "Do not cancel all PRE before POST by default"],
            ["POST_SKIP_IF_PRE_MATCHED", "FALSE", "Allow POST when PRE matched unless changed later"],
        ],
        "README": [
            ["Dogbot strategy Excel README"],
            ["Set Strategies.enabled to TRUE/FALSE to activate or deactivate a strategy."],
            ["Add Conditions rows linked by strategy_id. Same group = AND; multiple groups = OR."],
            ["LIMIT uses price_mode LIMIT_LTP, LIMIT_THEO or LIMIT_THEO_FUNC."],
            ["SP_MOC uses Betfair SP without a limit. Do not fill sp_limit_price."],
            ["SP_LOC is represented but currently rejected for Gruss Excel until trigger support is confirmed."],
            ["Do not use BSP/result/profit variables as live conditions; they are BACKTEST_ONLY."],
            ["Stake rule: positive stake below 1 EUR floors to 1 EUR; max default is 5 EUR."],
        ],
    }
    _write_strategy_workbook_guarded(
        target,
        rows,
        overwrite_config=overwrite_config,
        allow_template=allow_template,
        min_strategies=min_strategies,
    )
    return target


def export_strategy_slots_to_excel(
    slots: Iterable[Slot],
    path: Path | str | None = None,
    *,
    migration_report_path: Path | str = DEFAULT_STRATEGY_EXCEL_MIGRATION_REPORT_PATH,
    overwrite_config: bool = False,
    allow_template: bool = False,
    min_strategies: int = 20,
) -> dict[str, int]:
    """Export the currently loaded Python strategy slots into the Excel schema.

    Conditions are derived from the current registry metadata and known simple
    strategy patterns. Conditions that cannot be represented safely keep the row
    disabled with a review warning.
    """
    target = default_strategy_excel_export_path() if path is None else Path(path)

    strategy_rows: list[list[Any]] = []
    condition_rows: list[list[Any]] = []
    migration_rows: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    exported_count = 0
    active_count = 0
    disabled_count = 0
    warning_count = 0

    for priority, slot in enumerate(slots, start=1):
        exported_count += 1
        strategy_id = str(getattr(slot, "tag", "") or f"{slot.family}_{slot.slot}")
        name = str(getattr(slot, "strategy_name", "") or strategy_id)
        market_type = _slot_market_type(slot)
        side = _slot_side(slot)
        phase = str(getattr(slot, "execution_phase", EXECUTION_PHASE_POST) or EXECUTION_PHASE_POST).upper()
        order_mode, price_mode, sp_limit_price, sp_limit_variable, sp_limit_multiplier = _slot_order_fields(slot)
        price_limit_variable = sp_limit_variable
        price_limit_factor = sp_limit_multiplier
        limit_style = str(getattr(getattr(slot, "limit_style", None), "value", getattr(slot, "limit_style", "")) or "")
        price_for_bounds = str(getattr(slot, "price_for_bounds", "") or "")
        hyb_enabled = "TRUE" if order_mode == "HYB" else "FALSE"
        hyb_fallback_to_sp_moc = "TRUE"
        edge_env = str(getattr(slot, "edge_env", "") or "")
        max_runner_stake_env = str(getattr(slot, "max_runner_stake_env", "") or "")
        requires_mom45 = "TRUE" if bool(getattr(slot, "requires_mom45", False)) else "FALSE"
        bet_per_market = "TRUE" if bool(getattr(slot, "bet_per_market", False)) else "FALSE"
        strategy_group = str(getattr(slot, "strategy_group", "") or "")
        strategy_region = str(getattr(slot, "strategy_region", "") or "")
        strategy_signal = str(getattr(slot, "strategy_signal", "") or "")
        strategy_bucket = str(getattr(slot, "strategy_bucket", "") or "")
        stake_profile = str(getattr(slot, "stake_profile", "") or _default_stake_profile(side))
        function_name = str(getattr(slot, "function_name", "") or "")
        declarative_conditions, conversion_warning, conversion_message = _slot_excel_conditions(slot)
        order_supported = order_mode in {"LIMIT", "SP_MOC", "HYB"}
        convertible = bool(declarative_conditions) and order_supported and not conversion_warning
        enabled = "TRUE" if convertible else "FALSE"
        migrated_conditions_count = 0
        warning = ""
        message = "converted"
        description_parts = [
            f"Exported from Python registry slot={getattr(slot, 'slot', '')}",
            f"group={getattr(slot, 'strategy_group', '') or ''}",
            f"signal={getattr(slot, 'strategy_signal', '') or ''}",
            f"bucket={getattr(slot, 'strategy_bucket', '') or ''}",
            f"edge_env={getattr(slot, 'edge_env', '') or ''}",
        ]
        if not order_supported:
            warning = "order_mode_not_represented"
            message = f"{order_mode} order mode needs manual review before Excel activation"
            description_parts.append("TODO manual review - partial conversion")
        elif conversion_warning:
            warning = conversion_warning
            message = conversion_message
            description_parts.append("TODO manual review - partial conversion")
        elif not declarative_conditions:
            warning = "complex_python_condition"
            message = "condition callable could not be converted safely"
            description_parts.append("TODO manual review - complex Python condition")

        if not convertible:
            disabled_count += 1
            warning_count += 1
        else:
            active_count += 1
        for condition in declarative_conditions:
            migrated_conditions_count += 1
            condition_rows.append(
                [
                    "TRUE" if _parse_bool(condition.get("enabled"), default=True) else "FALSE",
                    strategy_id,
                    condition.get("group", "1"),
                    condition.get("variable", ""),
                    condition.get("operator", ""),
                    condition.get("value", ""),
                    condition.get("description", "Migrated from Python strategy metadata"),
                ]
            )

        strategy_rows.append(
            [
                enabled,
                strategy_id,
                name,
                market_type,
                side,
                phase,
                order_mode,
                price_mode,
                "" if sp_limit_price is None else sp_limit_price,
                sp_limit_variable,
                "" if sp_limit_multiplier is None else sp_limit_multiplier,
                price_limit_variable,
                "" if price_limit_factor is None else price_limit_factor,
                limit_style,
                price_for_bounds,
                hyb_enabled,
                "default",
                hyb_fallback_to_sp_moc,
                edge_env,
                max_runner_stake_env,
                requires_mom45,
                bet_per_market,
                strategy_group,
                strategy_region,
                strategy_signal,
                strategy_bucket,
                stake_profile,
                priority,
                " | ".join(str(part) for part in description_parts if str(part).strip()),
                function_name,
            ]
        )
        migration_rows.append(
            {
                "timestamp": timestamp,
                "python_strategy_id": strategy_id,
                "excel_strategy_id": strategy_id,
                "name": name,
                "status": "converted" if convertible else ("partial_conversion" if migrated_conditions_count else "manual_review_required"),
                "enabled": enabled,
                "market_type": market_type,
                "side": side,
                "phase": phase,
                "migrated_conditions_count": migrated_conditions_count,
                "warning": warning,
                "message": message,
            }
        )

    workbook_rows = _strategy_workbook_rows(
        strategies=strategy_rows,
        conditions=condition_rows,
        readme_intro=[
            "This workbook was generated from the current Python strategy registry.",
            "Callable Python conditions without declarative metadata are exported disabled for manual review.",
        ],
    )
    _write_strategy_workbook_guarded(
        target,
        workbook_rows,
        overwrite_config=overwrite_config,
        allow_template=allow_template,
        min_strategies=min_strategies,
    )
    _write_migration_report(migration_rows, migration_report_path)
    return {
        "python_strategies_detected": exported_count,
        "excel_strategies_exported": exported_count,
        "active_strategies": active_count,
        "disabled_with_warning": disabled_count,
        "partial_conversions": sum(1 for row in migration_rows if row["status"] == "partial_conversion"),
        "manual_review_required": sum(1 for row in migration_rows if row["status"] == "manual_review_required"),
        "migration_warnings": warning_count,
    }


def _strategy_workbook_rows(
    *,
    strategies: list[list[Any]],
    conditions: list[list[Any]],
    readme_intro: list[str] | None = None,
) -> dict[str, list[list[Any]]]:
    readme_rows = [[line] for line in (readme_intro or [])]
    readme_rows.extend(
        [
            ["Dogbot strategy Excel README"],
            ["Set Strategies.enabled to TRUE/FALSE to activate or deactivate a strategy."],
            ["Add Conditions rows linked by strategy_id. Same group = AND; multiple groups = OR."],
            ["LIMIT uses price_mode LIMIT_LTP, LIMIT_THEO or LIMIT_THEO_FUNC."],
            ["SP_MOC uses Betfair SP without a limit. Do not fill sp_limit_price."],
            ["SP_LOC is represented but currently rejected for Gruss Excel until trigger support is confirmed."],
            ["Do not use BSP/result/profit variables as live conditions; they are BACKTEST_ONLY."],
            ["Stake rule: positive stake below 1 EUR floors to 1 EUR; max default is 5 EUR."],
        ]
    )
    return {
        "Strategies": [STRATEGIES_COLUMNS, *strategies],
        "Conditions": [CONDITIONS_COLUMNS, *conditions],
        "Variables": [VARIABLES_COLUMNS, *[list(row) for row in LIVE_VARIABLES]],
        "StakeProfiles": [
            STAKE_PROFILES_COLUMNS,
            ["FIXED_1", "FIXED", "1", "1", "1", "", "Fixed stake 1"],
            ["FIXED_2", "FIXED", "1", "2", "2", "", "Fixed stake 2"],
            ["BACK_STANDARD", "VARIABLE", "1", "5", "2", "1", "Standard BACK variable stake"],
            ["LAY_STANDARD", "VARIABLE", "1", "5", "2", "1", "Standard LAY variable stake"],
        ],
        "GlobalSettings": [
            GLOBAL_SETTINGS_COLUMNS,
            ["PRE_LADDER_STEPS", "52,38,26,16", "Default PRE ladder steps"],
            ["POST_SEND_SECONDS", "12", "Informational V1 setting"],
            ["MIN_STAKE", "1", "Minimum real stake"],
            ["MAX_STAKE", "5", "Maximum real stake"],
            ["PRE_POST_INDEPENDENT", "TRUE", "PRE and POST can both run"],
            ["PRE_CANCEL_BEFORE_POST", "FALSE", "Do not cancel all PRE before POST by default"],
            ["POST_SKIP_IF_PRE_MATCHED", "FALSE", "Allow POST when PRE matched unless changed later"],
        ],
        "README": readme_rows,
    }


def _write_migration_report(rows: list[dict[str, Any]], path: Path | str) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MIGRATION_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _slot_market_type(slot: Slot) -> str:
    market_family = str(getattr(slot, "market_family", "") or "").upper()
    family = str(getattr(slot, "family", "") or "").upper()
    if market_family in {"WIN", "PLACE"}:
        return market_family
    if "PLACE" in family:
        return "PLACE"
    return "WIN"


def _slot_side(slot: Slot) -> str:
    side = getattr(slot, "side", "")
    return str(getattr(side, "value", side) or "").upper()


def _slot_order_fields(slot: Slot) -> tuple[str, str, float | None, str, Any]:
    mode = getattr(slot, "exec_mode", ExecMode.LIMIT_LTP)
    if mode == ExecMode.SP_MOC:
        return "SP_MOC", "SP_NO_LIMIT", None, "", None
    if mode == ExecMode.SP_LOC:
        sp_limit = _float_or_none(getattr(slot, "sp_limit", None))
        if sp_limit is not None:
            return "SP_LOC", "SP_LIMIT_FIXED", sp_limit, "", None
        return "SP_LOC", "SP_LIMIT_FROM_VARIABLE", None, str(getattr(slot, "sp_limit_variable", "") or ""), None
    if mode == ExecMode.HYB:
        return "HYB", "HYB_POLICY", None, "", None
    if str(getattr(slot, "price_mode", "") or "").upper() == "LIMIT_THEO_FUNC":
        variable = str(getattr(slot, "price_limit_variable", "") or "place_theorique")
        factor = getattr(slot, "price_limit_factor", "") or "DYNAMIC"
        return "LIMIT", "LIMIT_THEO_FUNC", None, variable, factor
    price_mode = "LIMIT_THEO" if getattr(slot, "sp_limit_fn", None) is not None else "LIMIT_LTP"
    sp_limit_variable = "place_theorique" if price_mode == "LIMIT_THEO" and _slot_market_type(slot) == "PLACE" else ""
    multiplier = _infer_sp_limit_multiplier(slot, sp_limit_variable)
    return "LIMIT", price_mode, None, sp_limit_variable, multiplier


def _default_stake_profile(side: str) -> str:
    return "LAY_STANDARD" if side == "LAY" else "BACK_STANDARD"


def _slot_excel_conditions(slot: Slot) -> tuple[list[dict[str, Any]], str, str]:
    conditions = getattr(slot, "excel_conditions", None)
    if conditions:
        normalized: list[dict[str, Any]] = []
        for condition in conditions:
            if not isinstance(condition, dict):
                return [], "invalid_declarative_condition", "slot excel_conditions contains a non-dict condition"
            normalized.append(dict(condition))
        return normalized, "", "converted from declarative metadata"
    return _derive_conditions_from_slot_metadata(slot)


def _infer_sp_limit_multiplier(slot: Slot, variable: str) -> float | None:
    fn = getattr(slot, "sp_limit_fn", None)
    if fn is None or not variable:
        return None
    try:
        ctx = RunnerCtx(
            market_id="migration",
            market_type=_slot_market_type(slot),
            selection_id=1,
            course_id="migration",
            ltp=3.0,
            place_theo=10.0,
            winbet=10.0,
        )
        value = _float_or_none(fn(ctx))
    except Exception:
        return None
    if value is None:
        return None
    return round(value / 10.0, 6)


def _condition(variable: str, operator: str, value: Any = "", description: str = "") -> dict[str, Any]:
    return {
        "enabled": "TRUE",
        "group": "1",
        "variable": variable,
        "operator": operator,
        "value": value,
        "description": description,
    }


def _derive_conditions_from_slot_metadata(slot: Slot) -> tuple[list[dict[str, Any]], str, str]:
    market_type = _slot_market_type(slot)
    region = str(getattr(slot, "strategy_region", "") or "").upper()
    signal = str(getattr(slot, "strategy_signal", "") or "").upper()
    bucket = str(getattr(slot, "strategy_bucket", "") or "").upper()
    strategy_id = str(getattr(slot, "tag", "") or "")
    price_variable = "win_price_ref" if market_type == "WIN" else "place_price_ref"
    conditions: list[dict[str, Any]] = [
        _condition("market_type", "=", market_type, "Python: ctx.market_type"),
    ]
    if region in {"UK", "ROW"}:
        conditions.append(_condition("region", "=", region, "Python: ctx.region"))
    if signal == "TRAP1":
        conditions.append(_condition("trap", "=", "1", "Python: ctx.trap == 1"))
    if signal == "TRAP8":
        conditions.append(_condition("trap", "=", "8", "Python: ctx.trap == 8"))
    if signal == "EV_PLACE" and market_type == "PLACE":
        conditions.append(_condition("place_theorique", ">", "1", "Python: _place_theo_ok(ctx)"))

    conditions.extend(_price_conditions_from_bucket(bucket, price_variable, signal, strategy_id))
    conditions.extend(_edge_conditions_for_slot(strategy_id, signal))
    conditions.extend(_mom45_conditions_from_bucket(bucket, signal))
    conditions.extend(_milestone_conditions_for_slot(strategy_id))

    if len(conditions) <= 1:
        return conditions, "complex_python_condition", "could not derive enough simple conditions from registry metadata"
    return conditions, "", "converted from Python registry metadata"


def _price_conditions_from_bucket(
    bucket: str,
    variable: str,
    signal: str,
    strategy_id: str,
) -> list[dict[str, Any]]:
    if not bucket:
        return []
    if "_PLUS" in bucket:
        value = (
            bucket.replace("WINBET_", "")
            .replace("PLACE_", "")
            .replace("_PLUS", "")
            .replace("_NEG", "")
            .replace("_POS", "")
        )
        try:
            threshold = float(value)
        except ValueError:
            return []
        operator = ">=" if signal == "EV_PLACE" else ">"
        return [_condition(variable, operator, _format_number(threshold), f"Python bucket {bucket}")]

    parts = bucket.replace("WINBET_", "").replace("PLACE_", "").split("_")
    if len(parts) < 2:
        return []
    try:
        lower = float(parts[0])
        upper = float(parts[1])
    except ValueError:
        return []

    if signal in {"TRAP8", "MOM45"}:
        lower_operator = ">"
        upper_operator = "<="
    else:
        lower_operator = ">="
        upper_operator = "<"
    if strategy_id == "BACK_WIN_543":
        lower_operator = ">"
        upper_operator = "<="
    return [
        _condition(variable, lower_operator, _format_number(lower), f"Python bucket {bucket}"),
        _condition(variable, upper_operator, _format_number(upper), f"Python bucket {bucket}"),
    ]


def _edge_conditions_for_slot(strategy_id: str, signal: str) -> list[dict[str, Any]]:
    thresholds = {
        "LAY_PLACE_502": ("<=", 0.0),
        "LAY_WIN_521": (">", 0.0),
        "LAY_WIN_522": (">", 0.10),
        "LAY_PLACE_523": ("<", 0.0),
        "LAY_WIN_401": (">=", 0.23),
        "LAY_WIN_402": (">=", 0.20),
        "BACK_WIN_411": ("<=", -0.12),
        "BACK_WIN_412": ("<=", -0.20),
        "BACK_WIN_413": ("<=", -0.40),
    }
    if strategy_id not in thresholds:
        return []
    operator, value = thresholds[strategy_id]
    return [_condition("ev_place", operator, _format_number(value), "Python: ev_place threshold")]


def _mom45_conditions_from_bucket(bucket: str, signal: str) -> list[dict[str, Any]]:
    if signal != "MOM45":
        return []
    if bucket.endswith("_NEG"):
        return [_condition("mom_45", "<", "-0.15", "Python: mom45 negative threshold")]
    if bucket.endswith("_POS"):
        return [_condition("mom_45", ">", "0.15", "Python: mom45 positive threshold")]
    return []


def _milestone_conditions_for_slot(strategy_id: str) -> list[dict[str, Any]]:
    if strategy_id in {"LAY_WIN_401", "LAY_WIN_402", "BACK_WIN_411", "BACK_WIN_412", "BACK_WIN_413"}:
        return [_condition("milestone", "=", "2", "Python: ctx.milestone == 2")]
    return []


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


def load_excel_strategy_slots(
    path: Path | str = DEFAULT_STRATEGY_EXCEL_PATH,
    *,
    report_path: Path | str = DEFAULT_STRATEGY_EXCEL_REPORT_PATH,
    write_report: bool = True,
) -> StrategyExcelLoadResult:
    workbook_path = Path(path)
    metadata = _workbook_file_metadata(workbook_path)
    issues: list[StrategyExcelIssue] = []
    if not workbook_path.exists():
        issue = StrategyExcelIssue("ERROR", "Workbook", "", "", "path", f"file_not_found={workbook_path}")
        result = StrategyExcelLoadResult([], [issue], 0, 0, 0, 0, {}, **metadata)
        if write_report:
            write_strategy_excel_report(result, report_path)
        raise StrategyExcelConfigError(result.errors)

    sheets = _read_xlsx(workbook_path)
    for sheet in SHEETS:
        if sheet not in sheets:
            issues.append(StrategyExcelIssue("ERROR", sheet, "", "", "sheet", "missing required sheet"))

    _validate_columns(sheets, "Strategies", STRATEGIES_COLUMNS, issues)
    _validate_columns(sheets, "Conditions", CONDITIONS_COLUMNS, issues)
    _validate_columns(sheets, "Variables", VARIABLES_COLUMNS, issues)
    _validate_columns(sheets, "StakeProfiles", STAKE_PROFILES_COLUMNS, issues)
    _validate_columns(sheets, "GlobalSettings", GLOBAL_SETTINGS_COLUMNS, issues)
    if issues:
        result = StrategyExcelLoadResult([], issues, 0, 0, 0, 0, {}, **metadata)
        if write_report:
            write_strategy_excel_report(result, report_path)
        raise StrategyExcelConfigError(result.errors)

    variable_rows = _rows_as_dicts(sheets["Variables"])
    variable_meta = {
        _norm_text(row.get("variable")): {
            "availability": _norm_text(row.get("availability")).upper(),
            "type": _norm_text(row.get("type")).lower(),
            "enabled": _parse_bool(row.get("enabled"), default=True),
        }
        for row in variable_rows
        if _norm_text(row.get("variable"))
    }
    stake_rows = _rows_as_dicts(sheets["StakeProfiles"])
    stake_profiles = {
        _norm_text(row.get("stake_profile")): row
        for row in stake_rows
        if _norm_text(row.get("stake_profile"))
    }
    global_settings = {
        _norm_text(row.get("key")): str(row.get("value") or "").strip()
        for row in _rows_as_dicts(sheets["GlobalSettings"])
        if _norm_text(row.get("key"))
    }

    strategy_rows = _rows_as_dicts(sheets["Strategies"])
    strategy_summary = _strategy_workbook_summary(strategy_rows)
    if "Strategy_Editor" in sheets:
        editor_rows = _rows_as_dicts(sheets.get("Strategy_Editor", []))
        editor_ids = {
            _norm_text(row.get("strategy_id"))
            for row in editor_rows
            if _norm_text(row.get("strategy_id"))
        }
        strategy_ids_in_sheet = {
            _norm_text(row.get("strategy_id"))
            for row in strategy_rows
            if _norm_text(row.get("strategy_id"))
        }
        if editor_ids and editor_ids != strategy_ids_in_sheet:
            only_editor = sorted(editor_ids - strategy_ids_in_sheet)
            only_technical = sorted(strategy_ids_in_sheet - editor_ids)
            issues.append(
                StrategyExcelIssue(
                    "ERROR",
                    "Strategy_Editor",
                    "",
                    "",
                    "strategy_id",
                    "Strategy_Editor and Strategies strategy_id sets differ "
                    f"editor_count={len(editor_ids)} strategies_count={len(strategy_ids_in_sheet)} "
                    f"only_editor={','.join(only_editor[:10])} "
                    f"only_strategies={','.join(only_technical[:10])}",
                )
            )
    condition_rows = _rows_as_dicts(sheets["Conditions"])
    conditions_by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in condition_rows:
        if not any(str(value or "").strip() for value in row.values()):
            continue
        sid = _norm_text(row.get("strategy_id"))
        conditions_by_strategy.setdefault(sid, []).append(row)

    seen_ids: set[str] = set()
    slots: list[Slot] = []
    disabled_count = 0
    strategies_read = 0
    for index, row in enumerate(strategy_rows, start=2):
        if not any(str(value or "").strip() for value in row.values()):
            continue
        strategies_read += 1
        sid = _norm_text(row.get("strategy_id"))
        enabled = _parse_bool(row.get("enabled"), default=False)
        if not sid:
            issues.append(_issue("Strategies", index, sid, "strategy_id", "strategy_id is required"))
            continue
        if sid in seen_ids:
            issues.append(_issue("Strategies", index, sid, "strategy_id", "strategy_id must be unique"))
            continue
        seen_ids.add(sid)
        if not enabled:
            disabled_count += 1
            continue

        market_type = _norm_text(row.get("market_type")).upper()
        side_text = _norm_text(row.get("side")).upper()
        phase = _norm_text(row.get("phase")).upper()
        order_mode = _norm_text(row.get("order_mode")).upper()
        price_mode = _norm_text(row.get("price_mode")).upper()
        stake_profile = _norm_text(row.get("stake_profile"))

        if market_type not in {"WIN", "PLACE"}:
            issues.append(_issue("Strategies", index, sid, "market_type", "market_type must be WIN or PLACE"))
        if side_text not in {"BACK", "LAY"}:
            issues.append(_issue("Strategies", index, sid, "side", "side must be BACK or LAY"))
        if phase not in {EXECUTION_PHASE_PRE, EXECUTION_PHASE_POST}:
            issues.append(_issue("Strategies", index, sid, "phase", "phase must be PRE or POST"))
        if order_mode not in {"LIMIT", "SP_MOC", "SP_LOC", "HYB"}:
            issues.append(_issue("Strategies", index, sid, "order_mode", "order_mode must be LIMIT, SP_MOC, SP_LOC or HYB"))
        if stake_profile not in stake_profiles:
            issues.append(_issue("Strategies", index, sid, "stake_profile", "stake_profile does not exist"))
        if order_mode == "LIMIT" and price_mode not in {"LIMIT_LTP", "LIMIT_THEO", "LIMIT_THEO_FUNC"}:
            issues.append(_issue("Strategies", index, sid, "price_mode", "LIMIT requires LIMIT_LTP, LIMIT_THEO or LIMIT_THEO_FUNC"))
        if order_mode == "LIMIT" and price_mode == "LIMIT_THEO_FUNC" and not _norm_text(row.get("function_name")):
            issues.append(_issue("Strategies", index, sid, "function_name", "LIMIT_THEO_FUNC requires function_name"))
        if order_mode == "SP_MOC" and price_mode != "SP_NO_LIMIT":
            issues.append(_issue("Strategies", index, sid, "price_mode", "SP_MOC requires SP_NO_LIMIT"))
        if order_mode == "HYB" and price_mode != "HYB_POLICY":
            issues.append(_issue("Strategies", index, sid, "price_mode", "HYB requires HYB_POLICY"))
        if order_mode == "SP_MOC" and _norm_text(row.get("sp_limit_price")):
            issues.append(_issue("Strategies", index, sid, "sp_limit_price", "SP_MOC must not define sp_limit_price"))
        if order_mode == "SP_LOC":
            issues.append(_issue("Strategies", index, sid, "order_mode", "sp_loc_not_supported_yet"))

        active_conditions = [
            condition
            for condition in conditions_by_strategy.get(sid, [])
            if _parse_bool(condition.get("enabled"), default=False)
        ]
        if not active_conditions:
            issues.append(_issue("Conditions", "", sid, "strategy_id", "active strategy has no active conditions"))
        for condition_index, condition in _indexed_conditions(condition_rows, sid):
            if not _parse_bool(condition.get("enabled"), default=False):
                continue
            variable = _norm_text(condition.get("variable"))
            operator = _norm_text(condition.get("operator")).upper()
            if variable not in variable_meta:
                issues.append(_issue("Conditions", condition_index, sid, "variable", f"unknown variable {variable!r}"))
                continue
            availability = variable_meta[variable]["availability"]
            if availability != "LIVE":
                issues.append(_issue("Conditions", condition_index, sid, "variable", f"variable {variable!r} is {availability}, not LIVE"))
            if operator not in ALLOWED_CONDITION_OPERATORS:
                issues.append(_issue("Conditions", condition_index, sid, "operator", f"unsupported operator {operator!r}"))

        if any(issue.strategy_id == sid and issue.severity == "ERROR" for issue in issues):
            continue

        slot = _slot_from_excel_row(row, sid, market_type, side_text, phase, order_mode, price_mode, active_conditions)
        slots.append(slot)

    for condition_index, condition in enumerate(condition_rows, start=2):
        if not any(str(value or "").strip() for value in condition.values()):
            continue
        sid = _norm_text(condition.get("strategy_id"))
        if sid and sid not in seen_ids:
            issues.append(_issue("Conditions", condition_index, sid, "strategy_id", "condition references unknown strategy_id"))

    result = StrategyExcelLoadResult(
        slots=slots,
        issues=issues,
        strategies_read=strategies_read,
        active_count=len(slots),
        disabled_count=disabled_count,
        conditions_read=len([row for row in condition_rows if any(str(value or "").strip() for value in row.values())]),
        global_settings=global_settings,
        **metadata,
        **strategy_summary,
    )
    if write_report:
        write_strategy_excel_report(result, report_path)
    if result.errors:
        raise StrategyExcelConfigError(result.errors)
    return result


def write_strategy_excel_report(
    result: StrategyExcelLoadResult,
    path: Path | str = DEFAULT_STRATEGY_EXCEL_REPORT_PATH,
) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rows = result.issues or [
        StrategyExcelIssue(
            "OK",
            "Workbook",
            "",
            "",
            "load",
            f"active_strategies={result.active_count} first_strategy_ids={','.join(result.first_strategy_ids or [])}",
            "INFO",
        )
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for issue in rows:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "status": issue.status,
                    "sheet": issue.sheet,
                    "row": issue.row,
                    "strategy_id": issue.strategy_id,
                    "field": issue.field,
                    "message": issue.message,
                    "severity": issue.severity,
                    "excel_path": result.excel_path,
                    "absolute_excel_path": result.absolute_excel_path,
                    "file_modified_time": result.file_modified_time,
                    "file_size": result.file_size,
                    "strategies_count": result.strategies_read,
                    "active_strategies": result.active_count,
                    "disabled_strategies": result.disabled_count,
                    "first_strategy_ids": ",".join(result.first_strategy_ids or []),
                    "contains_LIMIT_THEO_FUNC_count": result.limit_theo_func_count,
                    "contains_function_name_count": result.function_name_count,
                    "functional_strategy_ids": ",".join(result.functional_strategy_ids or []),
                }
            )


def describe_load_result(result: StrategyExcelLoadResult, *, enabled: bool, path: Path | str) -> str:
    return (
        f"STRATEGY_EXCEL enabled={str(bool(enabled)).lower()} path={path} "
        f"absolute_path={result.absolute_excel_path or Path(path).resolve(strict=False)} "
        f"file_modified_time={result.file_modified_time or '-'} "
        f"file_size={result.file_size} "
        f"strategies_count={result.strategies_read} "
        f"active_strategies={result.active_count} disabled={result.disabled_count} "
        f"conditions={result.conditions_read} errors={len(result.errors)} "
        f"first_strategy_ids={','.join(result.first_strategy_ids or []) or '-'} "
        f"LIMIT_THEO_FUNC_count={result.limit_theo_func_count} "
        f"function_name_count={result.function_name_count} "
        f"functional_strategy_ids={','.join(result.functional_strategy_ids or []) or '-'}"
    )


def _slot_from_excel_row(
    row: dict[str, Any],
    sid: str,
    market_type: str,
    side_text: str,
    phase: str,
    order_mode: str,
    price_mode: str,
    conditions: list[dict[str, Any]],
) -> Slot:
    family = f"{side_text}_{market_type}"
    slot_number = _slot_number_from_strategy_id(sid)
    exec_mode = {
        "LIMIT": ExecMode.LIMIT_LTP,
        "SP_MOC": ExecMode.SP_MOC,
        "SP_LOC": ExecMode.SP_LOC,
        "HYB": ExecMode.HYB,
    }[order_mode]
    sp_limit = _float_or_none(row.get("sp_limit_price"))
    sp_limit_variable = _norm_text(row.get("price_limit_variable")) or _norm_text(row.get("sp_limit_variable"))
    price_limit_factor_raw = _norm_text(row.get("price_limit_factor"))
    function_name = _norm_text(row.get("function_name")).upper()
    sp_limit_multiplier = _float_or_none(row.get("price_limit_factor"))
    if sp_limit_multiplier is None:
        sp_limit_multiplier = _float_or_none(row.get("sp_limit_multiplier"))
    if sp_limit_multiplier is None:
        sp_limit_multiplier = 1.0
    sp_limit_fn: Callable[[RunnerCtx], float | None] | None = None
    if price_mode == "LIMIT_THEO":
        variable_name = sp_limit_variable or ("place_theorique" if market_type == "PLACE" else "winbet")
        sp_limit_fn = lambda ctx, name=variable_name, multiplier=sp_limit_multiplier: (
            None
            if _float_or_none(_ctx_variable(ctx, name)) is None
            else float(_float_or_none(_ctx_variable(ctx, name))) * float(multiplier)
        )
    elif price_mode == "SP_LIMIT_FIXED":
        sp_limit_fn = None
    elif price_mode == "SP_LIMIT_FROM_VARIABLE":
        sp_limit_fn = lambda ctx, name=sp_limit_variable, multiplier=sp_limit_multiplier: (
            None
            if _float_or_none(_ctx_variable(ctx, name)) is None
            else float(_float_or_none(_ctx_variable(ctx, name))) * float(multiplier)
        )

    condition = _build_condition(sid, conditions)
    slot = Slot(
        family=family,
        slot=slot_number,
        side=Side(side_text),
        condition=condition,
        exec_mode=exec_mode,
        limit_style=_limit_style_from_excel(row.get("limit_style")),
        price_for_bounds=_norm_text(row.get("price_for_bounds")) or ("PLACE_BSP_THEN_LTP" if market_type == "PLACE" else "WINBET"),
        bet_per_market=_parse_bool(row.get("bet_per_market"), default=True),
        sp_limit=sp_limit,
        sp_limit_fn=sp_limit_fn,
        tag=sid,
        market_family=market_type,
        strategy_group=_norm_text(row.get("strategy_group")) or "EXCEL",
        strategy_region=_norm_text(row.get("strategy_region")) or None,
        strategy_signal=_norm_text(row.get("strategy_signal")) or "EXCEL",
        strategy_bucket=_norm_text(row.get("strategy_bucket")) or price_mode,
        requires_mom45=_parse_bool(row.get("requires_mom45"), default=False)
        or any(_norm_text(condition.get("variable")) in {"mom_45", "mom45"} for condition in conditions),
        edge_env=_norm_text(row.get("edge_env")) or None,
        max_runner_stake_env=_norm_text(row.get("max_runner_stake_env")) or None,
        execution_phase=phase,
    )
    slot.strategy_source = "excel"
    slot.strategy_name = _norm_text(row.get("name"))
    slot.order_mode = order_mode
    slot.price_mode = price_mode
    slot.function_name = function_name
    slot.stake_profile = _norm_text(row.get("stake_profile"))
    slot.sp_limit_variable = sp_limit_variable
    slot.sp_limit_multiplier = sp_limit_multiplier
    slot.price_limit_variable = sp_limit_variable
    slot.price_limit_factor = price_limit_factor_raw if price_mode == "LIMIT_THEO_FUNC" else sp_limit_multiplier
    slot.hyb_enabled = _parse_bool(row.get("hyb_enabled"), default=(order_mode == "HYB"))
    slot.hyb_policy = _norm_text(row.get("hyb_policy")) or ("default" if order_mode == "HYB" else "")
    slot.hyb_fallback_to_sp_moc = _parse_bool(row.get("hyb_fallback_to_sp_moc"), default=True)
    _attach_runtime_strategy_metadata(slot, order_mode, price_mode, sp_limit_variable, price_limit_factor_raw, function_name, conditions)
    return slot


def _attach_runtime_strategy_metadata(
    slot: Slot,
    order_mode: str,
    price_mode: str,
    limit_base_variable: str,
    limit_factor: str,
    function_name: str,
    conditions: list[dict[str, Any]],
) -> None:
    active_conditions = [condition for condition in conditions if _parse_bool(condition.get("enabled"), default=True)]
    variables = {_norm_text(condition.get("variable")) for condition in active_conditions}
    descriptions = [_norm_text(condition.get("description")) for condition in active_conditions]
    use_price_filter = bool(variables & {"place_price_ref", "win_price_ref", "cote_place", "cote_win", "ltp", "winbet"})
    use_trap_filter = "trap" in variables or bool(variables & {"is_trap_1", "is_trap_8"})
    use_ev_filter = "ev_place" in variables
    use_custom_filter = any(description.lower().startswith("editor raw extra") for description in descriptions)
    use_mom45_filter = bool(variables & {"mom_45", "mom45"})
    active_filters = []
    if use_price_filter:
        active_filters.append("price")
    if use_trap_filter:
        active_filters.append("trap")
    if use_ev_filter:
        active_filters.append("ev")
    if use_mom45_filter:
        active_filters.append("mom45")
    if use_custom_filter:
        active_filters.append("custom")

    setattr(slot, "use_price_filter", use_price_filter)
    setattr(slot, "use_trap_filter", use_trap_filter)
    setattr(slot, "use_ev_filter", use_ev_filter)
    setattr(slot, "use_custom_filter", use_custom_filter)
    setattr(slot, "use_mom45_filter", use_mom45_filter)
    setattr(slot, "active_filters", active_filters)
    setattr(slot, "limit_price_mode", _runtime_limit_price_mode(order_mode, price_mode))
    setattr(slot, "limit_base_variable", limit_base_variable)
    setattr(slot, "limit_factor", "DYNAMIC" if price_mode == "LIMIT_THEO_FUNC" else limit_factor)
    setattr(slot, "limit_function_name", function_name)


def _runtime_limit_price_mode(order_mode: str, price_mode: str) -> str:
    if order_mode == "SP_MOC":
        return "NONE"
    if price_mode == "LIMIT_THEO_FUNC":
        return "FUNCTION_COEFF"
    if price_mode == "LIMIT_THEO":
        return "THEO_FACTOR"
    if price_mode == "LIMIT_LTP":
        return "MANUAL"
    if price_mode == "HYB_POLICY":
        return "HYB_POLICY"
    return price_mode


def _build_condition(strategy_id: str, condition_rows: list[dict[str, Any]]) -> Callable[[RunnerCtx], bool]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in condition_rows:
        group = _norm_text(row.get("group")) or "1"
        groups.setdefault(group, []).append(row)

    def condition(ctx: RunnerCtx) -> bool:
        condition_group_matched = ""
        matched_conditions_count = 0
        failed_condition = ""
        failed_condition_reason = ""
        for group, rows in groups.items():
            group_ok = True
            local_count = 0
            for row in rows:
                ok, reason = _evaluate_condition(ctx, row)
                if ok:
                    local_count += 1
                    continue
                group_ok = False
                if not failed_condition:
                    failed_condition = f"{row.get('variable')} {row.get('operator')} {row.get('value')}"
                    failed_condition_reason = reason
                break
            if group_ok:
                condition_group_matched = group
                matched_conditions_count = local_count
                break
        setattr(condition, "condition_group_matched", condition_group_matched)
        setattr(condition, "matched_conditions_count", matched_conditions_count)
        setattr(condition, "failed_condition", failed_condition)
        setattr(condition, "failed_condition_reason", failed_condition_reason)
        return bool(condition_group_matched)

    setattr(condition, "strategy_id", strategy_id)
    return condition


def _evaluate_condition(ctx: RunnerCtx, row: dict[str, Any]) -> tuple[bool, str]:
    variable = _norm_text(row.get("variable"))
    operator = _norm_text(row.get("operator")).upper()
    expected = row.get("value")
    actual = _ctx_variable(ctx, variable)
    if operator == "IS_EMPTY":
        return _is_empty(actual), ""
    if operator == "IS_NOT_EMPTY":
        return not _is_empty(actual), ""
    if actual is None:
        return False, f"missing_variable={variable}"
    if operator == "IS_TRUE":
        return _truthy(actual), ""
    if operator == "IS_FALSE":
        return not _truthy(actual), ""
    if operator in {">", ">=", "<", "<=", "BETWEEN"}:
        number = _float_or_none(actual)
        if number is None:
            return False, f"not_numeric={variable}"
        if operator == "BETWEEN":
            bounds = [_float_or_none(part) for part in str(expected or "").split(",")]
            if len(bounds) != 2 or bounds[0] is None or bounds[1] is None:
                return False, "invalid_between_value"
            return bounds[0] <= number <= bounds[1], ""
        target = _float_or_none(expected)
        if target is None:
            return False, "invalid_numeric_value"
        if operator == ">":
            return number > target, ""
        if operator == ">=":
            return number >= target, ""
        if operator == "<":
            return number < target, ""
        return number <= target, ""
    actual_text = _norm_compare(actual)
    if operator in {"=", "=="}:
        return actual_text == _norm_compare(expected), ""
    if operator == "!=":
        return actual_text != _norm_compare(expected), ""
    if operator in {"IN", "NOT_IN"}:
        values = {_norm_compare(part) for part in str(expected or "").split(",")}
        contains = actual_text in values
        return (contains if operator == "IN" else not contains), ""
    return False, f"unsupported_operator={operator}"


def _ctx_variable(ctx: RunnerCtx, variable: str) -> Any:
    name = _norm_text(variable)
    region = str(ctx.region or "").upper()
    market_type = str(ctx.market_type or "").upper()
    win_price_ref = ctx.winbet or ctx.base_win or ctx.ltp
    place_price_ref = ctx.bsp_place if ctx.bsp_place and ctx.bsp_place > 1.0 else ctx.ltp
    mapping = {
        "region": ctx.region,
        "country_code": ctx.region,
        "market_type": ctx.market_type,
        "trap": ctx.trap,
        "runner_name": getattr(ctx, "runner_name", None),
        "selection_id": ctx.selection_id,
        "market_id": ctx.market_id,
        "course_id": ctx.course_id,
        "distance_m": getattr(ctx, "distance_m", None),
        "runners_count": getattr(ctx, "runners_count", None),
        "countdown": ctx.secs_to_off if ctx.secs_to_off is not None else ctx.milestone,
        "ltp": ctx.ltp,
        "bb": ctx.bb,
        "bl": ctx.bl,
        "best_back": ctx.bb,
        "best_lay": ctx.bl,
        "win_price_ref": win_price_ref,
        "place_price_ref": place_price_ref,
        "cote_win": win_price_ref,
        "cote_place": place_price_ref,
        "ltp_win": ctx.base_win or ctx.winbet,
        "ltp_place": ctx.ltp,
        "winbet": ctx.winbet,
        "place_theorique": ctx.place_theo,
        "placetheorique_p": ctx.place_theo,
        "ev_place": ctx.ev_place,
        "ratio_place": getattr(ctx, "ratio_place", None),
        "partenjeuxplace_p": getattr(ctx, "partenjeuxplace_p", None),
        "mom_45": ctx.mom45,
        "mom45": ctx.mom45,
        "mom_30": ctx.d30,
        "mom_20": getattr(ctx, "mom20", None),
        "mom_15": getattr(ctx, "mom15", None),
        "milestone": ctx.milestone,
        "is_uk": region == "UK",
        "is_row": region == "ROW",
        "is_win": market_type == "WIN",
        "is_place": market_type == "PLACE",
        "is_pre": str(ctx.execution_phase).upper() == EXECUTION_PHASE_PRE,
        "is_post": str(ctx.execution_phase).upper() == EXECUTION_PHASE_POST,
        "is_trap_1": ctx.trap == 1,
        "is_trap_8": ctx.trap == 8,
    }
    return mapping.get(name)


def _validate_columns(
    sheets: dict[str, list[list[Any]]],
    sheet: str,
    required: list[str],
    issues: list[StrategyExcelIssue],
) -> None:
    rows = sheets.get(sheet) or []
    header = [_norm_text(value) for value in (rows[0] if rows else [])]
    for column in required:
        if sheet == "Strategies" and column in OPTIONAL_STRATEGIES_COLUMNS:
            continue
        if column not in header:
            issues.append(_issue(sheet, 1, "", column, "missing required column"))


def _rows_as_dicts(rows: list[list[Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    header = [_norm_text(value) for value in rows[0]]
    result = []
    for row in rows[1:]:
        result.append({header[index]: row[index] if index < len(row) else "" for index in range(len(header))})
    return result


def _indexed_conditions(rows: list[dict[str, Any]], strategy_id: str) -> Iterable[tuple[int, dict[str, Any]]]:
    for index, row in enumerate(rows, start=2):
        if _norm_text(row.get("strategy_id")) == strategy_id:
            yield index, row


def _issue(sheet: str, row: int | str, strategy_id: str, field: str, message: str) -> StrategyExcelIssue:
    return StrategyExcelIssue("ERROR", sheet, row, strategy_id, field, message)


def _slot_number_from_strategy_id(strategy_id: str) -> int:
    match = re.search(r"(\d+)$", strategy_id)
    if match:
        return int(match.group(1))
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(strategy_id))
    return 900000 + checksum % 99999


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "oui"}


def _limit_style_from_excel(value: Any) -> LimitStyle:
    text = _norm_text(value).upper()
    if text == "PASSIVE":
        return LimitStyle.PASSIVE
    return LimitStyle.AGGRESSIVE


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "oui"}


def _is_empty(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_compare(value: Any) -> str:
    return str(value or "").strip().lower()


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_xlsx(path: Path, sheets: dict[str, list[list[Any]]]) -> None:
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel_doc = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ET.register_namespace("", ns_main)
    ET.register_namespace("r", ns_rel_doc)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        overrides = [
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        ]
        for index in range(1, len(sheets) + 1):
            overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            + "".join(overrides)
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        workbook_sheets = []
        workbook_rels = []
        for index, name in enumerate(sheets, start=1):
            workbook_sheets.append(f'<sheet name="{_xml_escape(name)}" sheetId="{index}" r:id="rId{index}"/>')
            workbook_rels.append(
                f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
            )
            zf.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(sheets[name]))
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(workbook_sheets)
            + "</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(workbook_rels)
            + "</Relationships>",
        )


def _sheet_xml(rows: list[list[Any]]) -> str:
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(row, start=1):
            if value is None:
                continue
            ref = f"{_column_name(column_index)}{row_index}"
            text = str(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{_xml_escape(text)}</t></is></c>')
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>'
    )


def _read_xlsx(path: Path) -> dict[str, list[list[Any]]]:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    rel_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("rel:Relationship", rel_ns)}
        shared_strings = _read_shared_strings(zf)
        result: dict[str, list[list[Any]]] = {}
        for sheet in workbook.findall("main:sheets/main:sheet", ns):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{ns['rel']}}}id"]
            target = rel_map[rel_id]
            sheet_path = "xl/" + target if not target.startswith("/") else target.lstrip("/")
            result[name] = _read_sheet(zf, sheet_path, shared_strings)
        return result


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("main:si", ns):
        values.append("".join(node.text or "" for node in item.findall(".//main:t", ns)))
    return values


def _read_sheet(zf: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[Any]]:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[Any]] = []
    for row in root.findall("main:sheetData/main:row", ns):
        values: dict[int, Any] = {}
        for cell in row.findall("main:c", ns):
            ref = cell.attrib.get("r", "A1")
            column_index = _column_index(re.match(r"[A-Z]+", ref).group(0))  # type: ignore[union-attr]
            values[column_index] = _cell_value(cell, ns, shared_strings)
        if values:
            max_index = max(values)
            rows.append([values.get(index, "") for index in range(1, max_index + 1)])
    return rows


def _cell_value(cell: ET.Element, ns: dict[str, str], shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//main:t", ns))
    value_node = cell.find("main:v", ns)
    if value_node is None or value_node.text is None:
        return ""
    if cell_type == "s":
        try:
            return shared_strings[int(value_node.text)]
        except (ValueError, IndexError):
            return ""
    text = value_node.text
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _column_index(name: str) -> int:
    result = 0
    for char in name:
        result = result * 26 + (ord(char.upper()) - 64)
    return result


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
