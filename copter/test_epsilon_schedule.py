#!/usr/bin/env python3
"""Regression tests for task-local and realistic global epsilon schedules."""

from structures import scheduled_epsilon


def close(actual, expected):
    assert abs(actual - expected) < 1e-12, (actual, expected)


def test_phase_schedule_restarts():
    close(scheduled_epsilon(1.0, 0.05, 100, 81, 1, "phase"), 0.9905)
    close(scheduled_epsilon(1.0, 0.05, 100, 180, 100, "phase"), 0.05)


def test_global_schedule_continues_across_phase_boundary():
    before = scheduled_epsilon(1.0, 0.05, 100, 80, 80, "global")
    after = scheduled_epsilon(1.0, 0.05, 100, 81, 1, "global")
    close(before, 0.24)
    close(after, 0.2305)
    assert after < before


def test_schedule_floor_and_validation():
    close(scheduled_epsilon(1.0, 0.05, 100, 1000, 1000, "global"), 0.05)
    close(scheduled_epsilon(1.0, 0.05, 0, 1, 1, "phase"), 0.05)
    try:
        scheduled_epsilon(1.0, 0.05, 100, 1, 1, "invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid epsilon schedule was accepted")


if __name__ == "__main__":
    test_phase_schedule_restarts()
    test_global_schedule_continues_across_phase_boundary()
    test_schedule_floor_and_validation()
    print("epsilon schedule tests passed (3/3)")
