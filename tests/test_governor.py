"""Tests for the budget governor: delay, degrade, refuse."""

import unittest

from relay.governor import Decision, Governor, GovernorConfig


def gov(**kw):
    cfg = GovernorConfig(rpm=30, tpm=8000, rpd=1000, tpd=200_000,
                         degrade_at_day_fraction=0.9)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return Governor(cfg)


class MinuteLimitsTest(unittest.TestCase):
    def test_under_limit_allows(self):
        g = gov()
        self.assertEqual(g.acquire(1000).action, "allow")

    def test_rpm_reached_delays_until_window_slides(self):
        g = gov(rpm=2)
        g.acquire(10).action, g.acquire(10).action
        d = g.acquire(10)

        self.assertEqual(d.action, "wait")
        self.assertGreater(d.wait_s, 0)

    def test_tpm_reached_delays(self):
        g = gov(tpm=1000)
        g.acquire(900)
        d = g.acquire(900)

        self.assertEqual(d.action, "wait")


class DayLimitsTest(unittest.TestCase):
    def test_near_daily_budget_degrades(self):
        g = gov(rpd=10, degrade_at_day_fraction=0.8)
        for _ in range(8):
            g.acquire(10)
        self.assertEqual(g.acquire(10).action, "degrade")
        self.assertTrue(g.degraded)

    def test_daily_budget_exhausted_refuses(self):
        g = gov(rpd=3)
        for _ in range(3):
            g.acquire(10)
        d = g.acquire(10)

        self.assertEqual(d.action, "refuse")
        self.assertTrue(d.exhausted)
        self.assertIn("daily", d.reason)

    def test_tpd_exhausted_refuses(self):
        g = gov(tpd=100)
        g.record(100)
        self.assertEqual(g.acquire(10).action, "refuse")

    def test_recording_tokens_affects_day_and_minute(self):
        g = gov(tpd=100)
        g.acquire(10)
        g.record(95)
        self.assertEqual(g.acquire(10).action, "refuse")

    def test_degrade_returns_to_normal_on_new_day(self):
        g = gov(rpd=2)
        g.acquire(10); g.acquire(10)
        g.reset_day()
        self.assertEqual(g.acquire(10).action, "allow")
        self.assertFalse(g.degraded)


if __name__ == "__main__":
    unittest.main()
