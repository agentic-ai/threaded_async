#  Copyright 2024 Agentic.AI Corporation.
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
"""Utilities for tracking timeouts."""

import datetime
from typing import Optional


class TimeoutTimer:
  """A helper context for tracking timeouts."""

  def __init__(
      self,
      timeout: Optional[float]):
    """Create a new timer with an optional duration.

    Args:
      timeout: The timeout in seconds, or None for no timeout.
        If the timeout is positive, self.remaining will return the remaining
        time. If the timeout is negative or None, self.remaining will return
        the timeout as is.
    """
    self._timeout = timeout
    self._start_time: Optional[datetime.datetime] = None

  @property
  def remaining(self) -> Optional[float]:
    """Get the remaining seconds, or None if no duration was set."""
    assert self._start_time, 'Timer not started.'
    if self._timeout is None or self._timeout < 0:
      return self._timeout
    elapsed = (datetime.datetime.now() - self._start_time).total_seconds()
    return max(self._timeout - elapsed, 0.0)

  def __enter__(self):
    assert not self._start_time, (
      'Timer already started, cannot re-enter.')
    self._start_time = datetime.datetime.now()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass

