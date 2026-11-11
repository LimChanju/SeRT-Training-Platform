from __future__ import annotations

import math
import unittest

from v3_chan.ac_feedback.online_protocol import FreeSingleCrossingMonitor


class FreeSingleCrossingMonitorTests(unittest.TestCase):
    A = (-1.0, 0.0, 0.0)
    B = (1.0, 0.0, 0.0)
    CENTER = (0.0, 0.0, 0.0)
    AWAY = (0.0, 2.0, 0.0)

    @staticmethod
    def monitor() -> FreeSingleCrossingMonitor:
        return FreeSingleCrossingMonitor(
            center_position_world=(0.0, 0.0, 0.0),
            tangent_world=(4.0, 0.0, 0.0),
            half_span_m=1.0,
            gate_radius_m=0.1,
            interaction_radius_m=0.2,
        )

    def observe(
        self,
        monitor: FreeSingleCrossingMonitor,
        time_s: float,
        *,
        left=(0.0, 2.0, 0.0),
        right=(0.0, 2.0, 0.0),
        left_valid: bool = True,
        right_valid: bool = True,
    ):
        return monitor.observe(
            left_position_world=left,
            right_position_world=right,
            left_valid=left_valid,
            right_valid=right_valid,
            simulation_time_s=time_s,
        )

    def test_either_hand_and_either_direction_can_be_first(self) -> None:
        cases = (
            ("left", self.A, self.B, 1),
            ("left", self.B, self.A, -1),
            ("right", self.A, self.B, 1),
            ("right", self.B, self.A, -1),
        )
        for hand, origin, destination, sign in cases:
            with self.subTest(hand=hand, sign=sign):
                monitor = self.monitor()
                positions = {"left": self.AWAY, "right": self.AWAY}
                positions[hand] = origin
                self.observe(monitor, 0.0, **positions)
                positions[hand] = self.CENTER
                self.observe(monitor, 1.0, **positions)
                positions[hand] = destination
                update = self.observe(monitor, 2.0, **positions)

                self.assertEqual(update.total_crossing_count, 1)
                self.assertEqual(len(update.completions), 1)
                self.assertEqual(update.completions[0].hand, hand)
                self.assertEqual(update.completions[0].sign, sign)
                metrics = monitor.finalize()
                self.assertFalse(metrics.off_protocol)
                self.assertEqual(metrics.first_crossing_hand, hand)
                self.assertEqual(metrics.first_completed_traversal_sign, sign)

    def test_first_crossing_metrics_are_latched_at_completion(self) -> None:
        monitor = self.monitor()
        self.observe(monitor, 3.0, left=self.A)
        self.observe(monitor, 4.0, left=self.CENTER)
        self.observe(monitor, 5.0, left=self.B)
        metrics = monitor.finalize()

        self.assertEqual(metrics.crossing_count, 1)
        self.assertEqual(metrics.first_crossing_hand, "left")
        self.assertEqual(metrics.first_completed_traversal_sign, 1)
        self.assertAlmostEqual(metrics.first_start_time_s, 3.0)
        self.assertAlmostEqual(metrics.first_end_time_s, 5.0)
        self.assertAlmostEqual(metrics.first_path_length_m, 2.0)
        self.assertAlmostEqual(metrics.first_duration_s, 2.0)
        self.assertAlmostEqual(metrics.first_mean_speed_mps, 1.0)
        self.assertAlmostEqual(metrics.first_rms_path_deviation_m, 0.0)
        self.assertAlmostEqual(metrics.first_maximum_path_deviation_m, 0.0)
        self.assertTrue(metrics.first_interaction_band_entered)

    def test_started_now_is_only_the_initial_gate_arm_edge(self) -> None:
        monitor = self.monitor()
        awaiting = self.observe(monitor, 0.0)
        self.assertFalse(awaiting.left_started_now)
        self.assertFalse(awaiting.right_started_now)

        left_armed = self.observe(monitor, 1.0, left=self.A)
        self.assertTrue(left_armed.left_started_now)
        self.assertFalse(left_armed.right_started_now)
        left_dwell = self.observe(monitor, 2.0, left=self.A)
        self.assertFalse(left_dwell.left_started_now)

        right_armed = self.observe(monitor, 3.0, right=self.B)
        self.assertFalse(right_armed.left_started_now)
        self.assertTrue(right_armed.right_started_now)

        left_completed = self.observe(monitor, 4.0, left=self.B)
        self.assertEqual(left_completed.total_crossing_count, 1)
        self.assertFalse(left_completed.left_started_now)
        self.assertFalse(left_completed.right_started_now)

    def test_missing_first_crossing_uses_timestamp_sentinels(self) -> None:
        metrics = self.monitor().finalize()
        self.assertEqual(metrics.first_start_time_s, -1.0)
        self.assertEqual(metrics.first_end_time_s, -1.0)

    def test_gate_dwell_and_halfway_return_do_not_complete(self) -> None:
        monitor = self.monitor()
        self.observe(monitor, 0.0, left=self.A)
        self.observe(monitor, 0.1, left=(-0.95, 0.0, 0.0))
        self.observe(monitor, 0.2, left=(-0.3, 0.0, 0.0))
        returned = self.observe(monitor, 0.3, left=self.A)
        self.assertEqual(returned.total_crossing_count, 0)
        self.assertFalse(returned.completions)

        self.observe(monitor, 1.0, left=(-0.4, 0.0, 0.0))
        completed = self.observe(monitor, 2.0, left=self.B)
        self.assertEqual(completed.total_crossing_count, 1)
        self.assertEqual(completed.completions[0].sign, 1)

    def test_complete_return_is_second_crossing_and_excluded(self) -> None:
        monitor = self.monitor()
        self.observe(monitor, 0.0, left=self.A)
        self.observe(monitor, 1.0, left=self.B)
        second = self.observe(monitor, 2.0, left=self.A)

        self.assertEqual(second.total_crossing_count, 2)
        self.assertEqual(second.completions[0].sign, -1)
        metrics = monitor.finalize()
        self.assertTrue(metrics.off_protocol)
        self.assertIn("multiple_crossing", metrics.off_protocol_reasons)
        self.assertEqual(metrics.first_completed_traversal_sign, 1)

    def test_opposite_hand_completion_is_also_second_crossing(self) -> None:
        monitor = self.monitor()
        self.observe(monitor, 0.0, left=self.A, right=self.B)
        first = self.observe(monitor, 1.0, left=self.B, right=self.B)
        second = self.observe(monitor, 2.0, left=self.B, right=self.A)

        self.assertEqual(first.total_crossing_count, 1)
        self.assertEqual(first.completions[0].hand, "left")
        self.assertEqual(second.total_crossing_count, 2)
        self.assertEqual(second.completions[0].hand, "right")
        self.assertEqual(second.completions[0].sign, -1)
        self.assertIn("multiple_crossing", monitor.finalize().off_protocol_reasons)

    def test_simultaneous_completions_are_both_counted(self) -> None:
        monitor = self.monitor()
        self.observe(monitor, 0.0, left=self.A, right=self.A)
        update = self.observe(monitor, 1.0, left=self.B, right=self.B)

        self.assertEqual(update.total_crossing_count, 2)
        self.assertEqual([event.hand for event in update.completions], ["left", "right"])
        self.assertEqual([event.sign for event in update.completions], [1, 1])
        self.assertEqual(update.first_crossing_hand, "left")
        self.assertIn("multiple_crossing", monitor.finalize().off_protocol_reasons)

    def test_reset_at_start_discards_pre_start_crossing(self) -> None:
        monitor = self.monitor()
        self.observe(monitor, 0.0, left=self.A)
        self.observe(monitor, 1.0, left=self.B)
        self.assertEqual(monitor.finalize().crossing_count, 1)

        monitor.reset()
        self.observe(monitor, 10.0, right=self.B)
        self.observe(monitor, 11.0, right=self.A)
        metrics = monitor.finalize()
        self.assertEqual(metrics.crossing_count, 1)
        self.assertEqual(metrics.first_crossing_hand, "right")
        self.assertEqual(metrics.first_completed_traversal_sign, -1)

    def test_invalid_tracking_is_a_fail_closed_reason(self) -> None:
        monitor = self.monitor()
        self.observe(monitor, 0.0, left=self.A)
        self.observe(
            monitor,
            0.5,
            left=None,
            left_valid=False,
        )
        self.observe(monitor, 1.0, left=self.B)
        metrics = monitor.finalize()

        self.assertEqual(metrics.crossing_count, 1)
        self.assertEqual(metrics.left_invalid_sample_count, 1)
        self.assertEqual(metrics.right_invalid_sample_count, 0)
        self.assertEqual(metrics.invalid_sample_count, 1)
        self.assertIn(
            "tracking_invalid_during_crossing", metrics.off_protocol_reasons
        )

    def test_first_crossing_must_enter_interaction_band(self) -> None:
        monitor = self.monitor()
        detour = (
            self.A,
            (-0.8, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.8, 1.0, 0.0),
            self.B,
        )
        for index, position in enumerate(detour):
            self.observe(monitor, float(index), left=position)
        metrics = monitor.finalize()

        self.assertEqual(metrics.crossing_count, 1)
        self.assertFalse(metrics.first_interaction_band_entered)
        self.assertIn("interaction_band_not_entered", metrics.off_protocol_reasons)
        self.assertGreater(metrics.first_maximum_path_deviation_m, 0.9)

    def test_segment_crossing_band_counts_when_sample_skips_center(self) -> None:
        monitor = self.monitor()
        self.observe(monitor, 0.0, left=self.A)
        self.observe(monitor, 2.0, left=self.B)
        metrics = monitor.finalize()

        self.assertTrue(metrics.first_interaction_band_entered)
        self.assertNotIn("interaction_band_not_entered", metrics.off_protocol_reasons)
        self.assertTrue(math.isclose(metrics.first_mean_speed_mps, 1.0))

    def test_broad_side_gates_accept_offset_paths_in_both_directions(self) -> None:
        # These radii intentionally exceed the 0.20 m half span.  The endpoint
        # samples are 0.25 m off-axis, so endpoint-centered spheres would reject
        # them while the broad side-plane/cylinder gates must accept them.
        offset_a = (-0.25, 0.25, 0.0)
        offset_b = (0.25, 0.25, 0.0)
        for hand, origin, destination, sign in (
            ("left", offset_a, offset_b, 1),
            ("right", offset_b, offset_a, -1),
        ):
            with self.subTest(hand=hand, sign=sign):
                monitor = FreeSingleCrossingMonitor(
                    center_position_world=self.CENTER,
                    tangent_world=(1.0, 0.0, 0.0),
                    half_span_m=0.20,
                    gate_radius_m=0.35,
                    interaction_radius_m=0.30,
                )
                positions = {"left": self.AWAY, "right": self.AWAY}
                positions[hand] = origin
                self.observe(monitor, 0.0, **positions)
                positions[hand] = destination
                update = self.observe(monitor, 1.0, **positions)

                self.assertEqual(update.total_crossing_count, 1)
                self.assertEqual(update.completions[0].hand, hand)
                self.assertEqual(update.completions[0].sign, sign)
                self.assertFalse(monitor.finalize().off_protocol)


if __name__ == "__main__":
    unittest.main()
