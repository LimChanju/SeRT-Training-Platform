from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

import v3_chan.collect_ac_selective_smoothing_feedback as collector
from v3_chan.ac_feedback.config import load_config


def _call_tail(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _instruction_argument(call: ast.Call) -> ast.expr | None:
    if len(call.args) >= 2:
        return call.args[1]
    for keyword in call.keywords:
        if keyword.arg in {"text", "instruction"}:
            return keyword.value
    return None


def _literal_fragments(node: ast.AST) -> str:
    return "".join(
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    )


class FreeCrossingRuntimeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module_source = inspect.getsource(collector)
        cls.module_tree = ast.parse(cls.module_source)
        cls.run_trial_tree = ast.parse(
            textwrap.dedent(inspect.getsource(collector._run_trial))
        )

    def test_production_collector_does_not_construct_crossing_cue_visuals(self) -> None:
        imported_names = {
            alias.asname or alias.name
            for node in ast.walk(self.module_tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name == "CrossingCueVisuals"
        }
        constructors = [
            node
            for node in ast.walk(self.module_tree)
            if isinstance(node, ast.Call)
            and (
                _call_tail(node.func) == "CrossingCueVisuals"
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id in imported_names
                )
            )
        ]
        self.assertEqual(imported_names, set())
        self.assertEqual(constructors, [])

    def test_start_instruction_is_free_and_non_prescriptive(self) -> None:
        instruction_calls = [
            node
            for node in ast.walk(self.run_trial_tree)
            if isinstance(node, ast.Call) and _call_tail(node.func) == "_set_instruction"
        ]
        instruction_arguments = [
            argument
            for call in instruction_calls
            if (argument := _instruction_argument(call)) is not None
        ]
        start_arguments = [
            argument
            for argument in instruction_arguments
            if _literal_fragments(argument).strip().startswith("START")
        ]
        self.assertEqual(
            len(start_arguments),
            1,
            "collector must expose exactly one cardinal START instruction",
        )

        start_argument = start_arguments[0]
        start_text = _literal_fragments(start_argument)
        for required in ("편한 손", "한쪽", "반대쪽", "한 번", "자연스럽게"):
            with self.subTest(required=required):
                self.assertIn(required, start_text)

        self.assertFalse(
            any(isinstance(node, ast.JoinedStr) for node in ast.walk(start_argument)),
            "START instruction must not interpolate a scheduled factor",
        )
        scheduled_attributes = {
            node.attr
            for node in ast.walk(start_argument)
            if isinstance(node, ast.Attribute)
        }
        self.assertTrue(
            scheduled_attributes.isdisjoint({"crossing_hand", "direction", "speed"})
        )

        visible_instruction_text = "\n".join(
            _literal_fragments(argument) for argument in instruction_arguments
        ).casefold()
        for forbidden in (
            "지정 손",
            "slow",
            "fast",
            "초록",
            "빨강",
            "노랑",
            "guide",
            "BC 단독",
            "본래 행동",
            "BC+CBF",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.casefold(), visible_instruction_text)

        visible_scheduled_attributes = {
            node.attr
            for argument in instruction_arguments
            for node in ast.walk(argument)
            if isinstance(node, ast.Attribute)
        }
        self.assertTrue(
            visible_scheduled_attributes.isdisjoint(
                {"crossing_hand", "direction", "speed"}
            ),
            "participant-visible instructions must not interpolate a scheduled "
            "hand, direction, or speed",
        )

    def test_config_disables_visible_markers_and_prescribed_crossing(self) -> None:
        config = load_config()
        self.assertIs(
            config["direct_live_scenario"][
                "participant_visible_crossing_markers"
            ],
            False,
        )
        self.assertIs(config["crossing"]["participant_prescribed"], False)

    def test_session_startup_copy_does_not_describe_a_baseline_controller(self) -> None:
        run_collection = next(
            node
            for node in self.module_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_collection"
        )
        instruction_text = "\n".join(
            _literal_fragments(argument)
            for call in ast.walk(run_collection)
            if isinstance(call, ast.Call)
            and _call_tail(call.func) == "_set_instruction"
            and (argument := _instruction_argument(call)) is not None
        )
        self.assertIn("양손 컨트롤러를 가슴 앞 안전 위치에 유지", instruction_text)
        for forbidden in ("BC", "CBF", "시연", "본래 행동", "기준 행동"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, instruction_text)

    def test_first_crossing_gap_uses_only_selected_hand_and_time_window(self) -> None:
        samples = [
            (1, 0.9, 0.01, 0.02, True, True),
            (2, 1.0, 0.40, 0.09, True, True),
            (3, 1.5, 0.30, 0.01, True, True),
            (4, 1.7, 0.02, 0.03, False, True),
            (5, 2.0, 0.20, 0.50, True, True),
            (6, 2.1, 0.001, 0.001, True, True),
        ]
        self.assertAlmostEqual(
            collector._minimum_gap_for_first_crossing(
                samples,
                hand="left",
                start_time_s=1.0,
                end_time_s=2.0,
            ),
            0.20,
        )
        self.assertAlmostEqual(
            collector._minimum_gap_for_first_crossing(
                samples,
                hand="right",
                start_time_s=1.0,
                end_time_s=2.0,
            ),
            0.01,
        )

    def test_simultaneous_first_completions_mark_first_and_extra(self) -> None:
        self.assertEqual(
            collector._legacy_crossing_completion_markers(
                total_crossing_count=1, completions_now=1
            ),
            (True, False),
        )
        self.assertEqual(
            collector._legacy_crossing_completion_markers(
                total_crossing_count=2, completions_now=2
            ),
            (True, True),
        )
        self.assertEqual(
            collector._legacy_crossing_completion_markers(
                total_crossing_count=2, completions_now=1
            ),
            (False, True),
        )

    def test_crossing_clock_mapping_requires_exactly_one_step(self) -> None:
        clocks = {10: (1.0, 100), 11: (2.0, 200)}
        self.assertEqual(
            collector._resolve_crossing_event_step(
                clocks, timestamp_s=1.0, boundary="start"
            ),
            10,
        )
        with self.assertRaisesRegex(
            collector.CollectionAbort, "crossing_event_clock_unresolved"
        ):
            collector._resolve_crossing_event_step(
                clocks, timestamp_s=3.0, boundary="end"
            )
        with self.assertRaisesRegex(
            collector.CollectionAbort, "crossing_event_clock_unresolved"
        ):
            collector._resolve_crossing_event_step(
                {10: (1.0, 100), 11: (1.0, 200)},
                timestamp_s=1.0,
                boundary="start",
            )

    def test_logged_crossing_gaps_are_post_step_but_control_gap_is_causal(self) -> None:
        source = inspect.getsource(collector._perform_logged_step)
        self.assertIn(
            'info_after.get("left_end_effector_surface_gap_m", 10.0)',
            source,
        )
        self.assertIn(
            'info_after.get("right_end_effector_surface_gap_m", 10.0)',
            source,
        )
        self.assertIn("surface_gap_m=control_minimum_gap", source)
        self.assertIn("minimum_gap_m=control_minimum_gap", source)

    def test_post_query_page_is_dismissed_before_task_resume_instruction(self) -> None:
        source = inspect.getsource(collector._run_trial)
        query_saved = source.index("recorder.append_query(query)")
        page_dismissed = source.index(
            "feedback.dismiss_questionnaire()", query_saved
        )
        resume_instruction = source.index(
            "로봇이 남은 pick-and-place 작업을 완료 중입니다", page_dismissed
        )
        completion_loop = source.index(
            "not args.reset_after_query", resume_instruction
        )
        self.assertLess(query_saved, page_dismissed)
        self.assertLess(page_dismissed, resume_instruction)
        self.assertLess(resume_instruction, completion_loop)
        self.assertIn("다음 START 전에는 로봇에 다시 넣지 마세요", source)


if __name__ == "__main__":
    unittest.main()
