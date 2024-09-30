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
"""Tests for the control_inversion module."""

import asyncio
import time
from typing import List, TypeVar
import unittest

from threaded_async import control_inversion
from threaded_async import threaded_async

ReturnT = TypeVar('ReturnT')


class AsyncServer(control_inversion.Server):
  """A server that records the future without doing anything."""

  def __init__(self):
    """Create a new exec server."""
    super().__init__()
    self.futures: List[threaded_async.Future] = []

  def _handle_request(
      self,
      unused_request: control_inversion.ExecutionRequest[ReturnT],
      future: threaded_async.Future[ReturnT]):
    """Execute the indicated function."""
    self.futures.append(future)

  @property
  def requests_handled(self) -> int:
    """Return the number of requests handled."""
    return len(self.futures)


class SyncServer(AsyncServer):
  """A server that simply executes the request and returns the result."""

  def _handle_request(
      self,
      request: control_inversion.ExecutionRequest[ReturnT],
      future: threaded_async.Future[ReturnT]):
    """Execute the indicated function."""
    super()._handle_request(request, future)
    try:
      result = request.fun(*request.args, **request.kw_args)
      future.set_result(result)
    except Exception as e:  # pylint: disable=broad-except
      future.set_exception(e)


class ControlInversionTest(unittest.TestCase):
  """Tests for control inversion."""

  def test_access_timeouts(self):
    """Test accessing timeouts."""
    with SyncServer() as server:
      self.assertIsNone(server.event_loop_timeout)
      self.assertIsNotNone(server.event_loop_exit_timeout)
      server.event_loop_timeout = 600.0
      self.assertEqual(server.event_loop_timeout, 600.0)
      server.event_loop_exit_timeout = 1200.0
      self.assertEqual(server.event_loop_exit_timeout, 1200.0)

  def test_create_task(self):
    """Test running a coroutine on the server."""
    with SyncServer() as server:
      async def foo(x: int):
        return x + 1
      self.assertEqual(server.create_background_task(foo(3)).wait(1.0), 4)
      self.assertEqual(server.requests_handled, 0)

  def test_process_request(self):
    """Tests processing a request."""
    def foo(x: int):
      return x + 1

    with SyncServer() as server:
      async def request_foo(client: control_inversion.Client):
        return await client.execute(foo, 3)
      task = server.create_background_task(request_foo(server.create_client()))
      server.process()   # Process one request
      self.assertEqual(task.wait(1.0), 4)
      self.assertEqual(server.requests_handled, 1)

  def test_process_delayed(self):
    """Tests delayed processing of a request."""
    def foo(x: int):
      return x + 1

    with AsyncServer() as server:
      async def request_foo(client: control_inversion.Client):
        return await client.execute(foo, 3)
      task = server.create_background_task(request_foo(server.create_client()))

      server.process()   # Process one request
      self.assertEqual(server.requests_handled, 1)

      # No result available yet.
      with self.assertRaises(TimeoutError):
        task.wait(0.05)

      # Inject answer.
      server.futures[0].set_result(10)
      # Fetch answer from task.
      self.assertEqual(task.wait(0.05), 10)

  def test_process_many(self):
    """Tests processing many requests."""
    def foo(x: int):
      return x + 1

    async def foo_requester(client: control_inversion.Client):
      """Request 100 foos and sum them."""
      total = 0
      for i in range(100):
        total += await client.execute(foo, i)
      return total

    with SyncServer() as server:
      task = server.create_background_task(
        foo_requester(server.create_client()))
      for _ in range(100):
        self.assertTrue(server.process())
      self.assertFalse(server.process())
      self.assertEqual(task.wait(1.0), 5050)
      self.assertEqual(server.requests_handled, 100)

  def test_shutdown_during_task(self):
    """Tests shutting down the server while running an awaitable."""
    async def foo():
      await asyncio.sleep(1.0)
      return 1
    now = time.time()
    with SyncServer() as server:
      task = server.create_background_task(foo())
    elapsed = time.time() - now
    self.assertLess(elapsed, 1.0)
    with self.assertRaises(asyncio.CancelledError):
      task.wait()

  def test_shutdown_during_request(self):
    """Tests shutting down the server while running an awaitable."""
    async def foo(client):
      return await client.execute(lambda x: x + 1, 1)

    with AsyncServer() as server:
      task = server.create_background_task(foo(server.create_client()))
      server.process()

    with self.assertRaises(asyncio.CancelledError):
      task.wait()

  def test_process_long_running_operations(self):
    """Tests process return value on long-running operations."""
    async def sequential(client: control_inversion.Client):
      """Issue 100 sequential requests."""
      total = 0
      for _ in range(100):
        total += await client.execute(lambda: 1)
      return total

    with AsyncServer() as server:
      task = server.create_background_task(sequential(server.create_client()))
      for _ in range(100):
        self.assertTrue(server.process())
        for _ in range(10):
          # No new queue items until we process the current one.
          self.assertFalse(server.process())
        server.futures[-1].set_result(1)
      self.assertEqual(task.wait(1.0), 100)


if __name__ == '__main__':  # pragma no cover
  unittest.main()
