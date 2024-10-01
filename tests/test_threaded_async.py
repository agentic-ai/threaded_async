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
"""Tests for the threaded_async module."""

import asyncio
import threading
import time
import unittest

from threaded_async import threaded_async


class TestAsyncRunner(unittest.TestCase):
  """Tests for AsyncRunner."""

  def test_run_awaitable(self):
    """Test running a coroutine."""
    async def foo(x: int):
      await asyncio.sleep(0.01)
      return x + 1
    with threaded_async.AsyncRunner() as runner:
      self.assertEqual(runner.run(foo(3)), 4)

  def _assert_timeout_exception_string_contains(
      self, exception: Exception, method_name: str, timeout: float):
    """Check a timeout exception's message string.

    Args:
      exception: Exception to check.
      method_name: Name of the method to find in the message.
      timeout: Timeout in seconds that should be reported in the message.
    """
    self.assertIsInstance(exception, threaded_async.EventLoopTimeoutError)
    exception_message = str(exception)
    self.assertRegex(exception_message,
                     f'execution of .*{method_name}')
    self.assertIn(f'waiting {timeout:.2}', exception_message)
    self.assertRegex(exception_message,
                     f'(?ms).*Traceback:.*{method_name}')

  def test_run_timeout(self):
    """Test timing out while running a coroutine."""
    async def coroutine_that_times_out():
      await asyncio.sleep(0.1)

    timeout = 0.01
    with threaded_async.AsyncRunner() as runner:
      with self.assertRaises(threaded_async.EventLoopTimeoutError) as context:
        runner.run(coroutine_that_times_out(), timeout=timeout)
    self._assert_timeout_exception_string_contains(
      context.exception, coroutine_that_times_out.__name__, timeout)

  def test_in_runner_loop(self):
    """Test the in_runner_loop function."""
    with threaded_async.AsyncRunner() as runner:
      self.assertFalse(runner.in_runner_loop)
      in_loop = None
      def _check_in_loop():
        nonlocal in_loop
        in_loop = runner.in_runner_loop
      runner.call_in_loop(_check_in_loop)
      self.assertTrue(in_loop)

  def test_in_runner_loop_seperate_loop(self):
    """Test the in_runner_loop function when run in a different event loop."""
    with threaded_async.AsyncRunner() as runner:
      async def in_loop():
        return runner.in_runner_loop
      self.assertFalse(asyncio.run(in_loop()))

  def test_create_task(self):
    """Tests creating and waiting for tasks."""
    async def foo(x: int):
      await asyncio.sleep(0.01)
      return x + 1

    async def bar(x: int):
      await asyncio.sleep(0.01)
      return await foo(x + 1)

    with threaded_async.AsyncRunner() as runner:
      task = runner.create_task(bar(3))
      self.assertEqual(task.wait(1.0), 5)

  def test_task_done(self):
    """Tests the task done method."""
    with threaded_async.AsyncRunner() as runner:
      event = threaded_async.Event(runner)

      async def await_event():
        await event.as_async_event().wait()

      task = runner.create_task(await_event())
      self.assertFalse(task.done())
      event.set()
      # Call is_set to make sure event loop advances.
      self.assertTrue(event.is_set())
      self.assertTrue(task.done())

  @staticmethod
  def _start_blocking_task(runner: threaded_async.AsyncRunner,
                           block_for_seconds: float) -> (
      threaded_async.BackgroundTask[None]):
    """Start a blocking task.

    Args:
      runner: Runner to start the task on.
      block_for_seconds: Time to block for.

    Returns:
      Blocking task.
    """
    blocking = threading.Event()

    async def block():
      blocking.set()
      time.sleep(block_for_seconds)

    task = runner.create_task(block())
    blocking.wait()
    return task

  def test_wait_for_next_eventloop_iteration_using_call_in_loop_timeout(self):
    """Test wait for the next iteration using the default timeout."""
    timeout = 0.1
    with threaded_async.AsyncRunner(call_in_loop_timeout=timeout) as runner:
      task = self._start_blocking_task(runner, 0.5)
      with self.assertRaises(threaded_async.EventLoopTimeoutError) as context:
        runner.wait_for_next_eventloop_iteration()
      self._assert_timeout_exception_string_contains(
        context.exception, 'block', timeout)
      task.wait(timeout=None)

  def test_wait_for_next_eventloop_iteration_using_override_timeout(self):
    """Test wait for the next iteration using an override timeout."""
    with threaded_async.AsyncRunner(call_in_loop_timeout=5.0) as runner:
      task = self._start_blocking_task(runner, 0.5)
      timeout = 0.1
      with self.assertRaises(threaded_async.EventLoopTimeoutError) as context:
        runner.wait_for_next_eventloop_iteration(timeout=timeout)
      self._assert_timeout_exception_string_contains(
        context.exception, 'block', timeout)
      task.wait(timeout=None)

  def test_wait_for_next_eventloop_iteration_using_no_timeout(self):
    """Test wait for the next iteration using no timeout."""
    with threaded_async.AsyncRunner(call_in_loop_timeout=0.5) as runner:
      task = self._start_blocking_task(runner, 1.0)
      runner.wait_for_next_eventloop_iteration(timeout=None)
      task.wait()

  def test_task_wait_on_done(self):
    """Tests the task can be waited on more than once."""
    with threaded_async.AsyncRunner() as runner:
      async def do_nothing():
        pass
      task = runner.create_task(do_nothing())
      task.wait()
      self.assertTrue(task.done())
      task.wait()

  def test_exception_handling(self):
    """Tests handling exceptions."""
    async def foo():
      raise ValueError('foo')

    with threaded_async.AsyncRunner() as runner:
      task = runner.create_task(foo())
      with self.assertRaises(ValueError):
        task.wait(1.0)

  def test_wait_after_shutdown(self):
    async def foo():
      await asyncio.sleep(10.0)
    with threaded_async.AsyncRunner() as runner:
      task = runner.create_task(foo())
    with self.assertRaises(asyncio.CancelledError):
      task.wait()

  def test_reraise_background_errors(self):
    """Tests that background errors are reraised on context exit."""
    with self.assertRaises(threaded_async.ExceptionGroup) as c:
      with threaded_async.AsyncRunner() as runner:
        async def foo(i):
          raise ValueError(i)
        _ = [runner.create_task(foo(i)) for i in range(10)]
        runner.run(asyncio.sleep(0.01))
      error_group = c.exception
      self.assertEqual(
        {e.args[0] for e in error_group.exceptions},
        set(range(10)))
      for exception in error_group.exceptions:
        self.assertIn('foo', str(exception))

  def test_cancel_outstanding_tasks(self):
    """Tests that outstanding tasks are cancelled on context exit."""
    with threaded_async.AsyncRunner() as runner:
      t = runner.create_task(asyncio.sleep(10.0))
      runner.run(asyncio.sleep(0.01))
    with self.assertRaises(asyncio.CancelledError):
      t.wait()

  def test_cancel_task(self):
    """Tests that background tasks can be manually cancelled."""
    with threaded_async.AsyncRunner() as runner:
      t = runner.create_task(asyncio.sleep(10.0))
      t.cancel()
      before = time.time()
      with self.assertRaises(asyncio.CancelledError):
        t.wait()
      duration = time.time() - before
      self.assertLess(duration, 0.1)

  def test_call_in_loop_timeout(self):
    def function_that_times_out():
      time.sleep(0.1)

    timeout = 0.05
    with threaded_async.AsyncRunner(
        call_in_loop_timeout=timeout) as runner:
      with self.assertRaises(threaded_async.EventLoopTimeoutError) as context:
        runner.call_in_loop(function_that_times_out)
    self._assert_timeout_exception_string_contains(
      context.exception, function_that_times_out.__name__, timeout)

  def test_cancelled_error(self):
    """Tests that cancelled_errors are correctly handled."""
    async def raise_cancelled_error():
      raise asyncio.CancelledError()

    with threaded_async.AsyncRunner() as runner:
      t = runner.create_task(raise_cancelled_error())
      with self.assertRaises(asyncio.CancelledError):
        t.wait()

  def test_wait_repeatedly(self):
    """Tests that cancelled_errors are correctly handled."""
    async def go_sleep():
      await asyncio.sleep(0.1)
      return 'ok'

    with threaded_async.AsyncRunner() as runner:
      t = runner.create_task(go_sleep())
      with self.assertRaises(threaded_async.EventLoopTimeoutError):
        t.wait(0.01)
      with self.assertRaises(threaded_async.EventLoopTimeoutError):
        t.wait(0.01)
      self.assertEqual(t.wait(0.5), 'ok')

  def test_shutdown_stopped(self):
    """Test stopping an event loop in an orderly fashion."""
    async def busy_wait():
      while True:
        await asyncio.sleep(0.01)
    with threaded_async.AsyncRunner() as runner:
      self.assertEqual(runner.shutdown_type,
                       threaded_async.ShutdownType.NOT_SHUT_DOWN)
      task = runner.create_task(busy_wait())
      time.sleep(0.1)
      task.cancel()
    self.assertEqual(runner.shutdown_type,
                     threaded_async.ShutdownType.STOPPED)

  def test_shutdown_force(self):
    """Test force stopping a stuck event loop."""
    busy_waiting = threading.Event()
    async def busy_wait():
      busy_waiting.set()
      while True:
        time.sleep(0.01)

    with threaded_async.AsyncRunner(
        log_forced_shutdown_stack_trace=False) as runner:
      _ = runner.create_task(busy_wait())
      busy_waiting.wait()
      time.sleep(0.1)

    self.assertEqual(runner.shutdown_type,
                     threaded_async.ShutdownType.FORCE_STOPPED)

  def test_shutdown_force_failed(self):
    """Test force stopping a stuck event loop that camps the GIL."""
    with threaded_async.AsyncRunner(
        log_forced_shutdown_stack_trace=False) as runner:
      _ = self._start_blocking_task(runner, 5.0)
      time.sleep(0.1)

    self.assertEqual(runner.shutdown_type,
                     threaded_async.ShutdownType.FAILED_FORCE_STOPPED)

  def test_asyncio_task_canceled_on_shutdown(self):
    """Test that outstanding asyncio tasks are cancelled on shutdown."""
    cancelled = False
    async def sleep_forever():
      nonlocal cancelled
      try:
        while True:
          await asyncio.sleep(1.0)
      except asyncio.CancelledError:
        cancelled = True
        raise

    async def start_sleep_forever_task(event: threaded_async.Event):
      # Start a background task
      task = asyncio.create_task(sleep_forever())
      event.set()
      await asyncio.sleep(10.0)  # We're going to abort here
      task.cancel()  # So the cancel here never executes.
      await task

    with threaded_async.AsyncRunner() as runner:
      event = threaded_async.Event(runner)
      _ = runner.create_task(start_sleep_forever_task(event))
      event.wait(10.0)
    self.assertTrue(cancelled)

class TestEvent(unittest.TestCase):
  """Tests for threaded_async.Event."""

  def test_thread_to_thread(self):
    """Test signalling thread to thread."""
    with threaded_async.AsyncRunner() as runner:
      event = threaded_async.Event(runner)
      event.set()
      event.wait(0.5)

  def test_thread_to_loop(self):
    """Test signalling thread to loop."""
    with threaded_async.AsyncRunner() as runner:
      event = threaded_async.Event(runner)
      async def await_event():
        await event.as_async_event().wait()
      await_task = runner.create_task(await_event())
      event.set()
      await_task.wait(0.5)

  def test_loop_to_thread(self):
    """Test signalling loop to thread."""
    with threaded_async.AsyncRunner() as runner:
      event = threaded_async.Event(runner)
      runner.call_in_loop(event.as_async_event().set)
      event.wait(0.5)


class TestFuture(unittest.TestCase):
  """Tests for threaded_async.Future."""

  def test_thread_to_thread_result(self):
    """Test signalling thread to thread."""
    with threaded_async.AsyncRunner() as runner:
      future = threaded_async.Future[int](runner)
      future.set_result(1)
      self.assertEqual(future.result_wait(), 1)

  def test_thread_to_thread_exception(self):
    """Test signalling thread to thread."""
    with threaded_async.AsyncRunner() as runner:
      future = threaded_async.Future[int](runner)
      future.set_exception(ValueError())
      with self.assertRaises(ValueError):
        self.assertEqual(future.result_wait(), 1)

  def test_thread_to_loop_result(self):
    """Test signalling thread to loop."""
    with threaded_async.AsyncRunner() as runner:
      future = threaded_async.Future[int](runner)
      async def await_future():
        return await future.result()
      await_task = runner.create_task(await_future())
      future.set_result(1)
      self.assertEqual(await_task.wait(0.5), 1)

  def test_thread_to_loop_exception(self):
    """Test signalling thread to loop."""
    with threaded_async.AsyncRunner() as runner:
      future = threaded_async.Future[int](runner)
      async def await_future():
        return await future.result()
      await_task = runner.create_task(await_future())
      future.set_exception(ValueError())
      with self.assertRaises(ValueError):
        await_task.wait(0.01)

  def test_loop_to_thread_result(self):
    """Test signalling loop to thread."""
    with threaded_async.AsyncRunner() as runner:
      future = threaded_async.Future[int](runner)
      runner.call_in_loop(lambda: future.set_result(1))
      self.assertEqual(future.result_wait(), 1)

  def test_loop_to_thread_exception(self):
    """Test signalling loop to thread."""
    with threaded_async.AsyncRunner() as runner:
      future = threaded_async.Future[int](runner)
      runner.call_in_loop(
        lambda: future.set_exception(ValueError()))
      with self.assertRaises(ValueError):
        future.result_wait()


class TestQueue(unittest.TestCase):
  """Tests for threaded_async.Queue."""

  def test_thread_to_thread(self):
    """Test transferring information from thread to thread via runner."""
    with threaded_async.AsyncRunner() as runner:
      queue: threaded_async.Queue[int] = threaded_async.Queue(runner)
      queue.put_wait(3)
      self.assertEqual(queue.get_wait(), 3)

  def test_loop_to_loop(self):
    """Test transferring information from within the loop to within the loop."""
    with threaded_async.AsyncRunner() as runner:
      queue: threaded_async.Queue[int] = threaded_async.Queue(runner, maxsize=1)

      async def foo():
        await queue.put(3)
        await queue.put(4)

      async def bar():
        await queue.get()
        return await queue.get()

      foo_task = runner.create_task(foo())
      bar_task = runner.create_task(bar())
      self.assertEqual(bar_task.wait(1.0), 4)
      foo_task.wait(1.0)


if __name__ == '__main__':  # pragma: no cover
  unittest.main()
