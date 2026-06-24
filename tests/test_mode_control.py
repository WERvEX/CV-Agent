from __future__ import annotations

from unittest.mock import patch

import pytest

from cv_agent.interaction.mode_control import offer_mode_control
from cv_agent.interaction.types import SessionQuit


def test_ask_mode_can_switch_to_auto():
    with patch("cv_agent.interaction.mode_control.sys.stdin.isatty", return_value=True):
        with patch("cv_agent.interaction.mode_control.select_action", return_value="auto"):
            result = offer_mode_control("ask", round_num=1)
    assert result == "auto"


def test_auto_mode_timeout_continues_auto():
    with patch("cv_agent.interaction.mode_control.sys.stdin.isatty", return_value=True):
        with patch("cv_agent.interaction.mode_control._wait_for_key_or_timeout", return_value=None):
            result = offer_mode_control("auto", round_num=2, auto_timeout_seconds=10.0)
    assert result == "auto"


def test_auto_mode_key_a_switches_to_ask():
    with patch("cv_agent.interaction.mode_control.sys.stdin.isatty", return_value=True):
        with patch("cv_agent.interaction.mode_control._wait_for_key_or_timeout", return_value="a"):
            result = offer_mode_control("auto", round_num=2)
    assert result == "ask"


def test_auto_mode_key_q_quits():
    with patch("cv_agent.interaction.mode_control.sys.stdin.isatty", return_value=True):
        with patch("cv_agent.interaction.mode_control._wait_for_key_or_timeout", return_value="q"):
            with pytest.raises(SessionQuit):
                offer_mode_control("auto", round_num=2)


def test_ask_mode_save_option_invokes_callback():
    with patch("cv_agent.interaction.mode_control.sys.stdin.isatty", return_value=True):
        with patch(
            "cv_agent.interaction.mode_control.select_action",
            side_effect=["save", "ask"],
        ):
            saved: list[bool] = []
            result = offer_mode_control(
                "ask",
                round_num=1,
                on_save_checkpoint=lambda: saved.append(True),
            )
    assert saved == [True]
    assert result == "ask"
