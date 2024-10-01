# Copyright 2023 Agentic.AI Corporation
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
"""Tests for the async_utils module."""

import asyncio
import time
import unittest

from threaded_async import async_utils


async def wait_for_or_cancel_scaffold(
    time_until_complete: float,
    time_until_cancel: float,
    timeout: float) -> int:
  """Function used in tests to test wait_for_or_cancel.

  Args:
    time_until_complete: The time to wait before returning a result.
    time_until_cancel: The time to wait before setting the cancel event.
    timeout: The timeout for wait_for_or_cancel.

  Returns:
    The number 12.

  Raises:
    asyncio.CancelledError: If the cancel event is set before the awaitable
      has completed.
    asyncio.TiemoutError: If the set timeout is exceeded.
  """
  async def return_result() -> int:
    await asyncio.sleep(time_until_complete)
    return 12

  cancel_event = asyncio.Event()

  async def set_cancel_event():
    await asyncio.sleep(time_until_cancel)
    cancel_event.set()
    await asyncio.sleep(10)

  cancel_task = set_cancel_event()
  wait_for_or_cancel_task = async_utils.wait_for_or_cancel(
    return_result(), cancel_event, timeout)
  done, _ = await asyncio.wait([cancel_task, wait_for_or_cancel_task],
                               return_when=asyncio.FIRST_COMPLETED)
  return list(done)[0].result()


class GatherOrCancelTest(unittest.TestCase):
  """Tests for gather_or_cancel."""

  def test_await_success_before_cancel(self):
    """Tests awaiting a coroutine that is successful before cancel."""
    self.assertEqual(asyncio.run(wait_for_or_cancel_scaffold(0, 0.1, 0.2)), 12)

  def test_cancel_before_success(self):
    """Tests awaiting a coroutine that is canceled before completion."""
    with self.assertRaises(asyncio.CancelledError):
      asyncio.run(wait_for_or_cancel_scaffold(0.1, 0, 0.2))

  def test_timeout(self):
    """Tests awaiting a coroutine that times out before cancel / success."""
    with self.assertRaises(asyncio.TimeoutError):
      asyncio.run(wait_for_or_cancel_scaffold(0.2, 0.3, 0.1))

class RunOrRaiseTest(unittest.TestCase):
  """Tests for run_or_raise."""
  def sleep_and_check_caller(
      self, total_sleep: float, last_caller: str, expected_last_caller: str):
    """Sleeps and checks that last_caller is expected_last_caller."""
    time.sleep(total_sleep)
    self.assertEqual(last_caller, expected_last_caller)

  def test_coroutine_returns_first(self):
    """Tests that run_or_raise returns the result of the coroutine."""
    last_caller = ''
    coroutine_sleep_seconds = 0.01
    raise_coroutine_sleep_seconds = 0.02

    async def coroutine() -> int:
      nonlocal last_caller
      await asyncio.sleep(coroutine_sleep_seconds)
      last_caller = 'coroutine'
      return 42

    async def raise_coroutine():
      nonlocal last_caller
      await asyncio.sleep(raise_coroutine_sleep_seconds)
      last_caller = 'raise_coroutine'
      raise ValueError('test')

    self.assertEqual(
      asyncio.run(async_utils.run_or_raise(coroutine(), raise_coroutine())),
      42)

    # To be extra careful, we sleep for the total amount of time
    # that the coroutine and raise_coroutine coroutines sleep.
    total_sleep = coroutine_sleep_seconds + raise_coroutine_sleep_seconds
    self.sleep_and_check_caller(total_sleep, last_caller, 'coroutine')


  def test_raiser_returns_first_raise(self):
    """Tests that run_or_raise raises the exception of the raise_coroutine."""
    last_caller = ''
    coroutine_sleep_seconds = 0.02
    raise_coroutine_sleep_seconds = 0.01

    async def coroutine():
      nonlocal last_caller
      await asyncio.sleep(coroutine_sleep_seconds)
      last_caller = 'coroutine'
      return 12

    async def raise_coroutine():
      nonlocal last_caller
      await asyncio.sleep(raise_coroutine_sleep_seconds)
      last_caller = 'raise_coroutine'
      raise ValueError('test')

    with self.assertRaisesRegex(ValueError, 'test'):
      asyncio.run(async_utils.run_or_raise(coroutine(), raise_coroutine()))

    # To be extra careful, we sleep for the total amount of time
    # that the coroutine and raise_coroutine coroutines sleep.
    total_sleep = coroutine_sleep_seconds + raise_coroutine_sleep_seconds
    self.sleep_and_check_caller(total_sleep, last_caller, 'raise_coroutine')

  def test_raiser_returns_first_no_raise(self):
    """Tests that run_or_raise raises the exception of the raise_coroutine."""
    last_caller = ''
    coroutine_sleep_seconds = 0.02
    raise_coroutine_sleep_seconds = 0.01

    async def coroutine():
      nonlocal last_caller
      await asyncio.sleep(coroutine_sleep_seconds)
      last_caller = 'coroutine'
      return 42

    async def raise_coroutine():
      nonlocal last_caller
      last_caller = 'raise_coroutine'
      return

    self.assertEqual(
      asyncio.run(async_utils.run_or_raise(coroutine(), raise_coroutine())),
      42)

    # To be extra careful, we sleep for the total amount of time
    # that the coroutine and raise_coroutine coroutines sleep.
    total_sleep = coroutine_sleep_seconds + raise_coroutine_sleep_seconds
    # Since raise_coroutine returned without raising before coroutine, we
    # expect last_caller to be 'coroutine'.
    self.sleep_and_check_caller(total_sleep, last_caller, 'coroutine')


if __name__ == '__main__':  # pragma no cover
  unittest.main()
