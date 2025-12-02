#!/usr/bin/env python3
"""
Scenario 1: the lid is opened outside the 8:00–12:00 dose window.
Steps:
1. Wait for initialization to finish, then type 'y' and press Enter.
2. The script freezes time at 07:05 and triggers the smart_pillbox alert path.
"""

import sys
import time
from pathlib import Path
from datetime import datetime as real_datetime
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    class _Cv2Stub:
        CAP_V4L2 = None
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5

        def VideoCapture(self, *_, **__):
            raise RuntimeError("cv2 stub active: camera unavailable in tests.")

        def imwrite(self, *_, **__):
            raise RuntimeError("cv2 stub active: cannot write images in tests.")

    sys.modules["cv2"] = _Cv2Stub()

import smart_pillbox  # noqa: E402


class DummyButton:
    def is_button_pressed(self):
        return False


def _wait_for_confirmation():
    while True:
        answer = input("Init ready. Type 'y' then Enter to simulate an early lid opening: ").strip().lower()
        if answer in {"y", "yes"}:
            return
        if answer in {"n", "no", ""}:
            print("Aborted by user.")
            sys.exit(0)
        print("Please respond with y/yes or n/no.")


def _build_fixed_datetime(hour: int, minute: int, second: int = 0):
    class _Fixed(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2024, 1, 1, hour, minute, second, tzinfo=tz)

    return _Fixed


def _setup_tester():
    dummy_button = DummyButton()
    patchers = [
        mock.patch.object(
            smart_pillbox.ButtonCameraTester,
            "_init_button",
            return_value=dummy_button,
        ),
        mock.patch.object(
            smart_pillbox.ButtonCameraTester,
            "_init_status_led",
            return_value=None,
        ),
    ]
    for p in patchers:
        p.start()

    def _cleanup():
        for p in reversed(patchers):
            p.stop()

    tester = smart_pillbox.ButtonCameraTester(prescriptions=[])
    if tester.touch_detector is None:
        cleanup()
        sys.exit("Touch detector is unavailable. Please run this test on hardware with a working MPR121 module.")

    tester._set_led_mode = lambda mode, hold_seconds=0.0: print(f"[LED] mode={mode} hold={hold_seconds}s")

    def _speak_with_audio(text, voice="en"):
        print(f"[VOICE] {text}")
        smart_pillbox.send_voice_prompt(text, voice=voice)

    tester._speak = _speak_with_audio

    return tester, _cleanup


def _wait_for_lid_open(tester):
    print("Waiting for the physical lid to open... (press Ctrl+C to abort)")
    while True:
        tester._check_touch_state()
        if tester.last_touch_state == "unlocked":
            return
        time.sleep(0.1)


def main():
    tester, cleanup = _setup_tester()
    try:
        print("=== Opening outside the window (should raise an error) ===")
        _wait_for_confirmation()

        fake_datetime = _build_fixed_datetime(7, 5)
        with mock.patch.object(smart_pillbox, "datetime", fake_datetime):
            _wait_for_lid_open(tester)
            tester._check_touch_state()

        print("WARNING: Lid was opened outside the 8:00-noon window; red LED should be flashing.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()

