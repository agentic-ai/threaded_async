# Copyright 2024 Agentic.AI Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the timers module."""

from threaded_async import timers

import unittest
import time


class TestTimers(unittest.TestCase):
  """Tests for the Timer class."""

  def test_timer_with_timeout(self):
    """Test a timer with timeout."""
    timer = timers.TimeoutTimer(timeout=1.0)
    with timer:
      time.sleep(0.05)
      r1 = timer.remaining
      time.sleep(0.05)
      r2 = timer.remaining
    self.assertTrue(1.0 > r1 > r2,
                    f"Expected 1.0 > r1={r1} > r2={r2}")

  def test_timer_with_none_timeout(self):
    """Test a timer with None as timeout."""
    timer = timers.TimeoutTimer(None)
    with timer:
      r1 = timer.remaining
      r2 = timer.remaining
    self.assertTrue(r1 is r2 is None)

  def test_timer_with_negative_timeout(self):
    """Test a timer with a negative timeout."""
    timer = timers.TimeoutTimer(-1.0)
    with timer:
      r1 = timer.remaining
      r2 = timer.remaining
    self.assertTrue(r1 == r2 == -1.0)


