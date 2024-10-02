# threaded\_async
A Python library for running async code in background threads.

Python offers two primary forms of concurrency: threads (`threading`) and
asynchronous coroutines (`asyncio`). This library provides support for running
asynchronous coroutines in background threads and includes synchronization
primitives that simplfiy coordinating between threaded and async code.

Note that in most cases, threading and coroutines are used independently, as
they serve different purposes and solve different problems. Before using this
library, it is worth considering whether combining the two is actually necessary
in your scenario.

## Quick start

Install the `threaded_async` library using `pip install threaded_async`.

Event loops can be created on a background thread by instantiating an
`AsyncRunner` and entering its context. Coroutines can be scheduled on these
background threads by creating a `BackgroundTask`:

```python
import asyncio
import threaded_async

async def foo() -> int:
  """Sleep and return a number."""
  await asyncio.sleep(0.1)
  return 10

with threaded_async.AsyncRunner() as runner:
  # Deploy the coroutine as a task on the loop
  background_task = runner.create_task(foo())
  # Block the current thread waiting for the task to complete.
  print(background_task.wait())
```

A number of synchronization primitives are also provided for bidirectional
communication between threads and async code. These include `Event`, `Queue` and
`Future`. Below is an example of running an async worker that processes a queue
filled by the main thread:

```python
import threaded_async

async def increment_worker(
    in_queue: threaded_async.Queue[int],
    out_queue: threaded_async.Queue[int]):
  """Increment integers from in_queue and put them in out_queue."""
  while True:
    number = await in_queue.get()
    await out_queue.put(number + 1)

with threaded_async.AsyncRunner() as runner:
  in_queue = threaded_async.Queue[int](runner)
  out_queue = threaded_async.Queue[int](runner)
  background_task = runner.create_task(increment_worker(in_queue, out_queue))
  for i in range(10):
    in_queue.put_wait(i)
    print(out_queue.get_wait())
  background_task.cancel()
```

## Control inversion

In some scenarios (e.g., AI scripts controlling video games), it is useful to
be able to deploy async code that interacts with an API, such that the timing
of when API calls are fulfilled is under the precise control of another thread.

Consider the following example script:

```python
async def client_code(client: Stub):
  for i in range(10):
    print(f'Client got: {await client.increment(i)}')
```

Typically, calling the `increment` function would trigger server code that
computes the appropriate result, but we would like the server to decide when to
process client requests. We refer to this pattern as _control inversion_, since
instead of the client request triggering work on the server, the server triggers
the client by providing results to past requests.

This can be accomplished by using queues as described above, but
`threaded_async` provides a convenience `Server` and `Client` class to support
this use case.

```python
from threaded_async.control_inversion import ExecutionRequest
from threaded_async.threaded_async import Future

class Stub(threaded_async.Client):
  """The interface between async coroutine and main thread."""
  async def increment(self, number: int) -> int:
    return await self.execute(Stub.increment, number)

class MyServer(threaded_async.Server):
  """A server that processes increment requests."""

  def _handle_request(
      self, request: ExecutionRequest[int], future: Future[int]):
    if request.fun == Stub.increment:
      # Handle increment request.
      (number,) = request.args
      future.set_result(number + 1)
    else:
      assert False, f"Unknown function {request.fun}"

server = MyServer()
with server:
  client = Stub(server)
  server.create_background_task(client_code(client))
  for i in range(3):
    print('Processing new client requests')
    server.process()
```

This will output the following (assuming synchronized printing):
```
Processing new client requests
Client got: 1
Processing new client requests
Client got: 2
Processing new client requests
Client got: 3
```

The client code waits on the request to the server until the `server.process`
function is called, which provides results and allows the client code to resume
execution.


## Development

To work in the development environment, you will need python 3.8 and pipenv
installed on your system. The following commands can be used to download the
code, set up the environment and run tests.

```bash
git clone https://github.com/agentic-ai/threaded_async.git
cd threaded_async
pipenv sync --dev
pipenv shell
./presubmit.sh  # Run tests / lint / typecheck
```

Before submitting a pull request, please ensure `./presubmit.sh` completes
without errors.

## More information

Additional information, e.g., about error handling and shutdown behavior can
be found in the [cookbook](examples/cookbook.ipynb).

