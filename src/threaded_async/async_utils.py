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
"""General utility functions for working with async code."""

import asyncio
from typing import Any, Awaitable, Coroutine, Optional, TypeVar


T_co = TypeVar('T_co', covariant=True)


async def wait_for_or_cancel(
    awaitable: Awaitable[T_co],
    cancel_event: asyncio.Event,
    timeout: Optional[float]=None) -> T_co:
  """Wait for the awaitable or abort with cancelled error.

  Args:
    awaitable: The awaitable to gather.
    cancel_event: An event that can be set to cancel the awaitable.
    timeout: An optional timeout.

  Returns:
    The result of the awaitable.

  Raises:
    asyncio.CancelledError: If the cancel event is set before the awaitable
      has completed.
    asyncio.TimeoutError: If the set timeout is exceeded.
  """
  wait = asyncio.wait_for(awaitable, timeout=timeout)
  await_task: Optional[asyncio.Task] = None
  wait_for_cancel_event: Optional[asyncio.Task] = None
  try:
    await_task = asyncio.create_task(wait)
    wait_for_cancel_event = asyncio.create_task(cancel_event.wait())
    _ = await asyncio.wait(
      [await_task, wait_for_cancel_event],
      return_when=asyncio.FIRST_COMPLETED,
      timeout=timeout)
    if wait_for_cancel_event.done():
      await_task.cancel()
      wait_for_cancel_event.result()  # Propagate exceptions.
    return await await_task
  finally:
    if await_task:
      await_task.cancel()
    if wait_for_cancel_event:
      wait_for_cancel_event.cancel()


async def run_or_raise(
    coroutine: Coroutine[Any, Any, T_co],
    raise_coroutine: Coroutine) -> T_co:
  """Run coroutine and cancel and reraise if raise_coroutine raises."""
  task = asyncio.create_task(coroutine)
  raise_task = asyncio.create_task(raise_coroutine)
  try:
    done, _ = await asyncio.wait(
      (task, raise_task),
      return_when=asyncio.FIRST_COMPLETED)

    if task in done:
      raise_task.cancel()
      return await task
    elif raise_task in done:
      # If raise_task finished first, reraise its exception.
      exception = raise_task.exception()
      if exception:
        task.cancel()
        raise exception
      else:
        return await task
    else:
      assert False, 'No task finished.'
  finally:
    task.cancel()
    raise_task.cancel()
