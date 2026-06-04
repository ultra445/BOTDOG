from __future__ import annotations

import unittest

from dogbot.executor import Executor
from dogbot.staking import Side
from dogbot.strategies import RunnerCtx, Slot, build_registry, try_fire_slot


class OptionalMom45Tests(unittest.TestCase):
    def test_registry_marks_only_mom45_strategies_as_requiring_mom45(self) -> None:
        slots = build_registry()
        required = {slot.tag for slot in slots if slot.requires_mom45}

        self.assertEqual(
            required,
            {"LAY_PLACE_541", "LAY_PLACE_542", "BACK_WIN_543", "LAY_PLACE_544", "LAY_PLACE_545"},
        )
        self.assertTrue(all(slot.requires_mom45 for slot in slots if slot.strategy_signal == "MOM45"))
        self.assertTrue(all(not slot.requires_mom45 for slot in slots if slot.strategy_signal != "MOM45"))

    def test_missing_mom45_does_not_block_back_place_101(self) -> None:
        slot = _slot("BACK_PLACE_101")
        result = try_fire_slot(None, slot, _place_ctx(region="UK", mom45=None))

        self.assertIsNotNone(result)

    def test_missing_mom45_does_not_block_lay_place_301(self) -> None:
        slot = _slot("LAY_PLACE_301")
        result = try_fire_slot(None, slot, _place_ctx(region="ROW", mom45=None))

        self.assertIsNotNone(result)

    def test_missing_mom45_blocks_only_requires_mom45_slot_without_calling_condition(self) -> None:
        slot = Slot(
            family="LAY_PLACE",
            slot=999,
            side=Side.LAY,
            condition=lambda ctx: (_ for _ in ()).throw(RuntimeError("condition must not run")),
            requires_mom45=True,
        )
        ctx = _place_ctx(region="UK", mom45=None)

        self.assertIsNone(try_fire_slot(None, slot, ctx))
        self.assertEqual(Executor.__new__(Executor)._debug_evaluate_slot(slot, ctx), (False, "missing_mom45"))

    def test_declared_mom45_slots_are_skipped_cleanly_when_mom45_is_none(self) -> None:
        ctx = _place_ctx(region="UK", mom45=None, ltp=20.0, place_theo=10.0)

        for strategy_id in ("LAY_PLACE_541", "LAY_PLACE_542"):
            slot = _slot(strategy_id)
            with self.subTest(strategy_id=strategy_id):
                self.assertIsNone(try_fire_slot(None, slot, ctx))
                self.assertEqual(Executor.__new__(Executor)._debug_evaluate_slot(slot, ctx), (False, "missing_mom45"))


def _slot(strategy_id: str) -> Slot:
    return next(slot for slot in build_registry() if slot.tag == strategy_id)


def _place_ctx(
    *,
    region: str,
    mom45: float | None,
    ltp: float = 2.0,
    place_theo: float = 2.0,
) -> RunnerCtx:
    return RunnerCtx(
        market_id="place-1",
        market_type="PLACE",
        selection_id=1,
        course_id="course-1",
        ltp=ltp,
        milestone=2,
        secs_to_off=2.0,
        mom45=mom45,
        bb=max(1.01, ltp - 0.1),
        bl=ltp + 0.1,
        region=region,
        place_theo=place_theo,
    )


if __name__ == "__main__":
    unittest.main()
