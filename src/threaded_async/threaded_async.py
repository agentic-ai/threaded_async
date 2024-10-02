#  Copyright 2023 Agentic.AI Corporation.
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
"""Module for running async event loops in background threads."""

import abc
import asyncio
import contextlib
import ctypes
import dataclasses
import enum
import threading
import traceback
import warnings

from concurrent import futures

from typing import (
  Any, Awaitable, Callable, Generator, Generic, List, Optional, Sequence, Set,
  Type, TypeVar, Union, cast)

from threaded_async import timers


# We only need this import for the futures interface. I played around for a
# a while trying to get structural subtyping to work instead so we could avoid
# this dependency, but ultimately gave up. (leo)

R = TypeVar('R')


def _force_thread_stop(
    thread: threading.Thread, exception: Type[BaseException]):
  """Raises an exception on another thread."""
  ident = thread.ident
  if ident is None:
    raise ValueError('Thread has not been started.')
  ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
    ctypes.c_ulong(ident),
    ctypes.py_object(exception))
  if ret == 0:
    raise ValueError(f'Thread not found: id={ident}')
  elif ret > 1:
    raise SystemError('Failed to raise exception on thread.')


@contextlib.contextmanager
def _suppress_unawaited_warning() -> Generator[None, None, None]:
  """Suppress warning about coroutines never being awaited."""
  with warnings.catch_warnings():
    warnings.filterwarnings(
      'ignore', category=RuntimeWarning,
      message='coroutine .* was never awaited')
    warnings.filterwarnings(
      'ignore', category=ResourceWarning,
      message='^unclosed event loop.*')
    yield


class EventLoopTimeoutError(TimeoutError):
  """Raised when waiting on the event loop times out."""

  def __init__(
      self, callable_like: Union[Callable, Awaitable, asyncio.Task],
      timeout: Optional[float], stack: str = ''):
    """Format the exception.

    Args:
      callable_like: Object that timed out.
      timeout: Time spent waiting for callable_like.
      stack: Call stack that was used to create the object that timed out.
    """
    if isinstance(callable_like, asyncio.Task):
      callable_description = callable_like.get_name()
      callable_description, stack = callable_description.split('\n', maxsplit=1)
    else:
      callable_description = str(callable_like)

    timeout_string = (
      '<infinite>' if timeout is None else f'{round(timeout, 2):.2}')
    stack = f'\nTraceback:\n{stack}\n' if stack else ''
    super().__init__(
      f'Timed out waiting {timeout_string} seconds for execution of '
      f'{callable_description} to complete.{stack}')


# Constant that applies a default timeout.
USE_DEFAULT_TIMEOUT = -1.0


class BackgroundTask(Generic[R], abc.ABC):
  """An async task running in a background thread."""

  @abc.abstractmethod
  def wait(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> R:
    """Block and wait for result.

    Args:
      timeout: An optional timeout.

    Returns:
      The task result.

    Raises:
      EventLoopTimeoutError: If the task does not complete within the timeout.
      Exception: If the task raises an exception.
    """

  @abc.abstractmethod
  def done(self) -> bool:
    """Return whether the task is done."""

  @abc.abstractmethod
  def cancel(self):
    """Cancel the background task."""

@dataclasses.dataclass(frozen=True)
class ExceptionDetails:
  exception: Exception
  traceback: str


class ExceptionGroup(Exception):
  """Group of exceptions raised by background task."""

  def __init__(self):
    """Create a new background task error."""
    super().__init__()
    self._exception_details: List[ExceptionDetails] = []

  @property
  def exception_details(self) -> Sequence[ExceptionDetails]:
    """The exception details that have been recorded."""
    return tuple(self._exception_details)

  @property
  def exceptions(self) -> Sequence[Exception]:
    """The exceptions that have been recorded."""
    return tuple(e.exception for e in self.exception_details)

  @property
  def tracebacks(self) -> Sequence[str]:
    """The tracebacks associated with each recorded exception."""
    return tuple(e.traceback for e in self.exception_details)

  def record_exception(self, exception: Exception):
    """Add an exception as it is being raised."""
    self._exception_details.append(ExceptionDetails(
      exception, traceback.format_exc()))

  def __str__(self) -> str:
    """Stringify the combined exception."""
    return '\n'.join(self.tracebacks)


class ShutdownType(enum.IntEnum):
  """Returns how the event loop was shut down."""
  NOT_SHUT_DOWN = 0          # Event loop was not shut down
  STOPPED = 1                # Event loop was stopped normally
  FORCE_STOPPED = 2          # Event loop thread was forced to stop.
  FAILED_FORCE_STOPPED = 3   # Thread may still be running.
  UNKNOWN = 4                # Something else went wrong

  def force_stop_attempted(self) -> bool:
    """Returns True if the event loop was force stopped."""
    return self in (ShutdownType.FORCE_STOPPED,
                    ShutdownType.FAILED_FORCE_STOPPED)


# TODO: Figure out why this behaves differently if inheriting from
# BaseException.
class ForcedThreadShutdown(SystemExit):
  """Raised when the event loop thread is forced to stop."""


class AsyncRunner:
  """Runs an async event loop in a background thread.

  Attributes:
    call_in_loop_timeout: Default timeout for call_in_loop, that is, the maximum
      time the main thread will wait for the event loop when scheduling work in
      the loop. This can be overriden by passing an explicit timeout to the
      call_in_loop method.
    exit_timeout: Timeout when waiting for the event loop to exit. If None
      will block indefinitely.
  """

  def __init__(
      self,
      exit_timeout: Optional[float] = 1.0,
      call_in_loop_timeout: Optional[float] = None,
      debug: bool = False,
      log_forced_shutdown_stack_trace: bool = True):
    """Start the event loop in the background.

    Args:
      exit_timeout: Timeout when waiting for the event loop to exit.
      call_in_loop_timeout: Default timeout for synchronous operations that
        are called on the loop, e.g., task creation. This helps raise
        exceptions if the event loop becomes blocked.
      debug: Whether to run the event loop in debug mode, which will track
        tracebacks for all tasks.
      log_forced_shutdown_stack_trace: Whether to log forced thread shutdown
        exceptions. This should only be disabled if used in a context when a
        forced thread shutdown is expected (e.g in tests).
    """
    self.exit_timeout = exit_timeout
    self.call_in_loop_timeout = call_in_loop_timeout
    self._event_loop = asyncio.new_event_loop()
    self._event_loop.set_debug(debug)
    self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
    self._done = threading.Event()
    self._log_forced_shutdown_stack_trace = log_forced_shutdown_stack_trace
    self._tasks: Set[_BackgroundTask] = set()
    self._shutdown_type = ShutdownType.NOT_SHUT_DOWN

  def _get_timeout_or_default(self, timeout: Optional[float]) -> (
      Optional[float]):
    """Get a timeout in seconds, None or the default.

    Args:
      timeout: Timeout to use, None or USE_DEFAULT_TIMEOUT.

    Returns:
      If timeout is >= 0 or None, returns time otherwise returns
      a default timeout (call_in_loop_timeout).
    """
    assert timeout is None or timeout >= 0.0 or timeout == USE_DEFAULT_TIMEOUT
    return (self.call_in_loop_timeout
            if timeout == USE_DEFAULT_TIMEOUT else timeout)

  def _run_event_loop(self):
    """Run the event loop."""
    try:
      self._event_loop.run_forever()
    except ForcedThreadShutdown:  # pragma: no cover
      if self._log_forced_shutdown_stack_trace:
        raise

  def _register_task(self, task: '_BackgroundTask'):
    """Register a task."""
    self._tasks.add(task)

  def _register_task_done(self, task: '_BackgroundTask'):
    """Register a task."""
    self._tasks.discard(task)

  def __enter__(self) -> 'AsyncRunner':
    """Enter context in which event loop is run."""
    assert not self.is_running
    self._thread.start()
    # Run a dummy task to ensure the event loop is running.
    self.wait_for_next_eventloop_iteration(timeout=None)
    return self

  def _reraise_pending_bg_tasks(self):
    """Reraises errors from background tasks. """
    combined_exceptions = ExceptionGroup()
    for t in self._tasks:
      try:
        t.reraise()
      except asyncio.CancelledError:
        pass
      except ForcedThreadShutdown:
        pass  # This happens during force shutdown.
      except Exception as e:  # pylint: disable=broad-except
        combined_exceptions.record_exception(e)
    if combined_exceptions.exceptions:
      raise combined_exceptions

  def _finalize_shutdown(self, timeout: Optional[float]):
    """Attempt to force close the event loop if necessary and clean up.

    Args:
      timeout: Time to wait for the thread to shut down or None to wait
        indefinitely.
    """
    with contextlib.ExitStack() as exit_stack:
      if self._shutdown_type != ShutdownType.STOPPED:
        # Suppress async warnings on force shutdown.
        exit_stack.enter_context(_suppress_unawaited_warning())
      if self._thread.is_alive():
        self._shutdown_type = ShutdownType.FORCE_STOPPED
        _force_thread_stop(self._thread, ForcedThreadShutdown)
        self._thread.join(timeout=timeout)
      if self._thread.is_alive():
        self._shutdown_type = ShutdownType.FAILED_FORCE_STOPPED
      self._reraise_pending_bg_tasks()
      self._done.set()
      try:
        self._event_loop.close()
      except RuntimeError:
        if not self._shutdown_type.force_stop_attempted():
          raise

  def __exit__(self, exc_type, exc_value, tb):
    """Stops the event loop."""
    # Give the event loop a chance to catch up to finish any pending tasks.
    with timers.TimeoutTimer(timeout=self.exit_timeout) as deadline:
      try:
        try:
          self.wait_for_next_eventloop_iteration(
            timeout=deadline.remaining)
          for bg_task in list(self._tasks):
            bg_task.cancel()
          for asyncio_task in asyncio.all_tasks(loop=self._event_loop):
            asyncio_task.cancel()
          # Cancellation is only guaranteed to take effect after the loop runs
          # its next iteration.
          self.wait_for_next_eventloop_iteration(
            timeout=deadline.remaining)
          _ = self._event_loop.call_soon_threadsafe(self._event_loop.stop)
          self._thread.join(timeout=deadline.remaining)
          self._shutdown_type = ShutdownType.STOPPED
        except EventLoopTimeoutError:
          pass
        finally:
          self._finalize_shutdown(self.exit_timeout)
      finally:
        if self._shutdown_type == ShutdownType.NOT_SHUT_DOWN:
          self._shutdown_type = ShutdownType.UNKNOWN

  def _assert_running(self):
    """Assert that the event loop is running."""
    assert self.is_running, 'Event loop is not running.'

  @property
  def shutdown_type(self) -> ShutdownType:
    """Provides information as to whether there was an orderly shutdown."""
    return self._shutdown_type

  @property
  def in_runner_loop(self) -> bool:
    """Returns True if we are inside the async runner event loop."""
    try:
      return asyncio.get_running_loop() == self._event_loop
    except RuntimeError:
      return False

  def call_in_loop(
      self,
      fun: Callable[[], R],
      timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> R:
    """Runs callable in loop and returns result.

    Note that this does not deploy a coroutine to the loop, but simply
    executes the function asynchronously. This function is useful for
    instantiating, e.g., asyncio.Event objects which store their associated
    loop.

    Args:
      fun: The callable to run. To run a function with arguments use
        functools.partial or a lambda expression to bind the argument.
        Note that this is not meant for deploying coroutines on the loop.
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the constructor.

    Returns:
      The function return value.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    self._assert_running()
    if self.in_runner_loop:
      return fun()
    tracker: _ExecutionTracker[R] = (
      _ExecutionTracker(self._get_timeout_or_default, fun))
    def _wrapper():
      with tracker.track():
        tracker.set_result(fun())
    _ = self._event_loop.call_soon_threadsafe(_wrapper)
    return tracker.wait(timeout=self._get_timeout_or_default(timeout))

  def create_task(
      self, awaitable: Awaitable[R],
      task_creation_timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> (
        BackgroundTask[R]):
    """Create a background task and return it.

    Args:
      awaitable: Awaitable to wrap in a task.
      task_creation_timeout: An optional timeout in seconds for task creation.
        Note that this is not a timeout for task completion but for task
        creation. Set to None to wait indefinitely or USE_DEFAULT_TIMEOUT to use
        call_in_loop_timeout set in the constructor.

    Returns:
      Task.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    self._assert_running()
    task = _BackgroundTask.create(
      awaitable, self, timeout=self._get_timeout_or_default(
        task_creation_timeout))
    return task

  def run(self, awaitable: Awaitable[R],
          timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> R:
    """Run an awaitable in the event loop, block and return the result.

    Args:
      awaitable: Awaitable to execute on the event loop.
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the constructor.

    Returns:
      Result of the awaitable.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    self._assert_running()
    with timers.TimeoutTimer(timeout=timeout) as deadline:
      return (self.create_task(
        awaitable, task_creation_timeout=deadline.remaining)
              .wait(timeout=deadline.remaining))

  def wait_for_next_eventloop_iteration(
      self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT):
    """Blocks until the next event loop iteration.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the constructor.

    Raises:
      EventLoopTimeoutError: If the event loop doesn't respond within the
        specified timeout.
    """
    wait_timeout = self._get_timeout_or_default(timeout)
    try:
      self.run(asyncio.sleep(0.0), timeout=wait_timeout)
    except EventLoopTimeoutError as error:
      task = asyncio.current_task(loop=self._event_loop)
      if task:
        raise EventLoopTimeoutError(task, wait_timeout) from error
      raise error

  @property
  def is_running(self) -> bool:
    """Returns True if the loop is alive and not being shut down."""
    return self._thread.is_alive() and not self._done.is_set()


class Event:
  """An asyncio event that can be set from an outside thread."""

  def __init__(self, runner: 'AsyncRunner',
               timeout: Optional[float] = USE_DEFAULT_TIMEOUT):
    """Create an Event that may be used for signaling with runner tasks.

    Args:
      runner: Runner to use the event in.
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in AsyncRunner's
        constructor.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    super().__init__()
    self._runner = runner
    self._event = runner.call_in_loop(asyncio.Event, timeout=timeout)

  def set(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT):
    """Set the event.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in AsyncRunner's
        constructor.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    self._runner.call_in_loop(self._event.set, timeout=timeout)

  def is_set(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> bool:
    """Returns whether the event is set.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in AsyncRunner's
        constructor.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    return self._runner.call_in_loop(self._event.is_set, timeout=timeout)

  def wait(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT):
    """Wait for the event from outside the loop.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in AsyncRunner's
        constructor.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    self._runner.run(self._event.wait(), timeout=timeout)

  def as_async_event(self) -> asyncio.Event:
    """Return the asyncio event."""
    return self._event


ValueT = TypeVar('ValueT')


class Future(Generic[ValueT]):
  """A future that can be set / awaited inside or outside the loop."""

  def __init__(self, runner: 'AsyncRunner'):
    """Create a new future."""
    self._concurrent_future: futures.Future = futures.Future()
    self._runner = runner
    self._done_event = Event(runner)

  def set_result(self, result: ValueT):
    """Set the result of the future.

    This call will error if a result or exception has already been set.

    Args:
      result: The value that should be retrieved upon calling
        result or result_wait.
    """
    self._concurrent_future.set_result(result)
    self._done_event.set()

  def set_exception(self, exception: Exception):
    """Set the exception of the future.

    This call will error if a result or exception has already been set.

    Args:
      exception: The exception that should be raised upon calling
        result or result_wait.
    """
    self._concurrent_future.set_exception(exception)
    self._done_event.set()

  def done(self) -> bool:
    """Return whether the future is done."""
    return self._concurrent_future.done()

  async def result(self) -> ValueT:
    """Asynchronously await the future."""
    await self._done_event.as_async_event().wait()
    return self._concurrent_future.result()

  def result_wait(
      self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> ValueT:
    """Block thread until the future is complete.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the constructor.

    Raises:
      EventLoopTimeoutError: If the future does not complete within the timeout.
      Exception: If the future raises an exception.
    """
    return self._runner.run(self.result(), timeout=timeout)


class Queue(Generic[R]):
  """A queue that can coordinate between an AsyncRunner and thread."""

  def __init__(self, runner: 'AsyncRunner', maxsize: int=0):
    """Initialize the queue."""
    self._runner = runner
    self._maxsize = maxsize
    self._async_queue: Optional[asyncio.Queue] = None
    self._lock = threading.Lock()

  def _create_asyncio_queue(self) -> asyncio.Queue:
    """Create the requested asyncio queue."""
    return asyncio.Queue(self._maxsize)

  def _init_queue(self):
    """Create the queue on the runner's thread."""
    with self._lock:
      if not self._async_queue:
        self._async_queue = self._runner.call_in_loop(
          self._create_asyncio_queue, timeout=None)

  @property
  def _queue(self) -> asyncio.Queue:
    """If not already created, create the queue."""
    if not self._async_queue:
      self._init_queue()
    assert self._async_queue is not None
    return self._async_queue

  def empty(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> bool:
    """Return whether the queue is empty.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the runner's
        constructor.

    Returns:
      Whether the queue is empty.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    return self._runner.call_in_loop(self._queue.empty, timeout=timeout)

  def full(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> bool:
    """Return whether the queue is full.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the runner's
        constructor.

    Returns:
      Whether the queue is full.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    return self._runner.call_in_loop(self._queue.full, timeout=timeout)

  def qsize(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> int:
    """Return the size of the queue.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the runner's
        constructor.

    Returns:
      Size of the queue.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    return self._runner.call_in_loop(self._queue.qsize, timeout=timeout)

  async def get(self) -> R:
    """Remove and return an item from the queue."""
    return cast(R, await self._queue.get())

  async def put(self, item: R):
    """Put an item on the queue, waiting until room is available"""
    await self._queue.put(cast(Any, item))

  def get_wait(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> R:
    """Block, remove and return an item from the queue.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the runner's
        constructor.

    Returns:
      Item from the queue.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    return self._runner.run(self.get(), timeout=timeout)

  def put_wait(self, item: R, timeout: Optional[float] = USE_DEFAULT_TIMEOUT):
    """Put an item into the queue.

    Args:
      item: Item to add to the queue.
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use call_in_loop_timeout set in the runner's
        constructor.

    Raises:
      EventLoopTimeoutError: If the function does not complete within the
        timeout.
    """
    _ = self._runner.run(self.put(item), timeout=timeout)


class _BackgroundTask(BackgroundTask[R]):
  """Implementation of background tasks."""

  def __init__(self, awaitable: Awaitable[R], runner: 'AsyncRunner'):
    """Internal constructor. Use the "create" method to create a task."""
    self._awaitable = awaitable
    self._runner = runner
    self._async_task: Optional[asyncio.Task] = None
    self._exec_tracker: _ExecutionTracker[R] = _ExecutionTracker(
      # pylint: disable=protected-access
      self._runner._get_timeout_or_default, awaitable)

  @property
  def _stack(self) -> str:
    """Stack that was used to create this object."""
    return self._exec_tracker.stack

  @classmethod
  def create(cls,
             awaitable: Awaitable[R],
             runner: 'AsyncRunner',
             timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> (
               BackgroundTask[R]):
    """Create a background task using the runner and return it."""
    bg_task = cls(awaitable, runner)
    bg_task._async_task = runner.call_in_loop(
      # pylint: disable=protected-access
      bg_task._create_task,
      timeout=timeout)
    # pylint: disable=protected-access
    runner._register_task(bg_task)
    return bg_task

  def _create_task(self) -> asyncio.Task:
    """Create the task that wraps the awaitable associated with this object."""
    return asyncio.create_task(
      self._run(),
      name=(f'{self._awaitable}\n'
            f'{self._stack}'))

  async def _run(self):
    """Await the task result."""
    with self._exec_tracker.track():
      self._exec_tracker.set_result(await self._awaitable)

  def reraise(self):
    """Reraise any exceptions that have occurred on the task."""
    self._exec_tracker.reraise()
    # During force shutdown the exception is sometimes not recorded in the
    # tracker, so we also check the task itself.
    if self._async_task and self._async_task:
      try:
        if e := self._async_task.exception():
          raise e
      except asyncio.InvalidStateError:
        # Tasks can enter invalid states during force shutdowns.
        pass

  def wait(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT) -> R:
    """Block and wait for result."""
    assert self._async_task is not None, (
      'Cannot wait for task that has not been started. '
      'Use the "create" method to create a task.')
    if self._runner.in_runner_loop:
      raise RuntimeError('Cannot wait for task in runner loop.')
    if not self._exec_tracker.is_done and not self._runner.is_running:
      raise RuntimeError('Cannot wait for task when runner is not running.')
    try:
      result = self._exec_tracker.wait(timeout=timeout)
    finally:
      if self._exec_tracker.is_done:
        self._runner._register_task_done(self)  # pylint: disable=protected-access
    return result

  def done(self) -> bool:
    """Return whether the task is done."""
    return self._exec_tracker.is_done

  def cancel(self, timeout: Optional[float] = USE_DEFAULT_TIMEOUT):
    """Cancel the background task."""
    if self._async_task is None:
      raise RuntimeError(
        'Cannot cancel task that has not been started. '
        'Use the "create" method to create a task.')
    self._runner.call_in_loop(self._async_task.cancel, timeout=timeout)


class _ExecutionTracker(Generic[R]):
  """Context manager for tracking execution of a function.

  TODO: This can probably be rewritten as a Future.

  Used as follows:

  Thread 1:
    with _ExecTracker() as tracker:
      ... # Do something
      tracker.set_result(result)

  Thread 2:
    result = tracker.wait(timeout=1.0)
  """

  def __init__(
      self, get_timeout_or_default: (
        Callable[[Optional[float]], Optional[float]]),
      callable_like: Union[Callable[[], R], Awaitable[R]]):
    """Create a new tracker.

    Args:
      get_timeout_or_default: Callable that given a timeout, None or
        USE_DEFAULT_TIMEOUT returns either the provided timeout, None or
        a default timeout respectively.
      callable_like: The callable being tracked. This is used to provide
        context if it times out.
    """
    self._done = threading.Event()
    self._result: List[R] = []
    self._result_set = False
    self._exception: Optional[BaseException] = None
    self._get_timeout_or_default = get_timeout_or_default
    self._callable_like = callable_like
    self._stack = ''.join(traceback.format_list(traceback.extract_stack())[:-1])

  @property
  def stack(self) -> str:
    """Stack that was used to create the object being tracked."""
    return self._stack

  def set_result(self, result: R):
    """Set the result."""
    if self._result_set:
      raise RuntimeError('Result already set.')
    self._result_set = True
    self._result.append(result)

  @contextlib.contextmanager
  def track(self):
    """Context manager for tracking executions."""
    try:
      yield self
    except Exception as e:  # pylint: disable=broad-except
      self._exception = e
    except asyncio.CancelledError as e:
      # Note that CancelledError does not inherit from Exception.
      self._exception = e
    finally:
      self._done.set()

  @property
  def is_done(self) -> bool:
    """Returns True if the function has completed."""
    return self._done.is_set()

  def reraise(self):
    """Non-blocking call to reraise any exceptions raised in the context."""
    if self._exception:
      raise self._exception

  def wait(self, timeout: Optional[float]) -> R:
    """Wait for the result.

    Args:
      timeout: An optional timeout in seconds, None to wait indefinitely or
        USE_DEFAULT_TIMEOUT to use the timeout returned by
        get_timeout_or_default specified on construction
        (typically AsyncRunner.call_in_loop_timeout).
    """
    wait_timeout = self._get_timeout_or_default(timeout)
    if not self._done.wait(timeout=wait_timeout):
      raise EventLoopTimeoutError(self._callable_like, wait_timeout,
                                  stack=self.stack)
    if self._exception is not None:
      raise self._exception
    if not self._result_set:
      raise asyncio.CancelledError()
    return self._result[0]
