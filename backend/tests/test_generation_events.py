"""Tests for the in-process generation_events bus.

The bus is the linchpin of the /events/generations SSE endpoint:
services/history.py publishes here on every status change, and the
SSE route fans events out to every subscriber. If the bus loses,
duplicates, or leaks subscribers, the browser's Send button regresses
back to the 6-connection-per-origin cliff that this PR set out to
avoid.

Run with: python -m pytest backend/tests/test_generation_events.py -v
"""

import asyncio
import pytest

from backend import generation_events


@pytest.fixture(autouse=True)
def _reset_bus_between_tests():
    """Drop all subscribers so each test sees a clean bus."""
    generation_events.bus._subscribers.clear()
    yield
    generation_events.bus._subscribers.clear()


def _drain(q: asyncio.Queue):
    """Return the next message from q without awaiting it."""
    assert not q.empty()
    return q.get_nowait()


def test_subscribe_returns_a_queue_and_registers_a_subscriber():
    generation_events.bus.subscribe()
    assert generation_events.bus.subscriber_count == 1


def test_subscribe_multiple_records_all_of_them():
    generation_events.bus.subscribe()
    generation_events.bus.subscribe()
    generation_events.bus.subscribe()
    assert generation_events.bus.subscriber_count == 3


def test_unsubscribe_removes_the_subscriber():
    q = generation_events.bus.subscribe()
    assert generation_events.bus.subscriber_count == 1
    generation_events.bus.unsubscribe(q)
    assert generation_events.bus.subscriber_count == 0


def test_unsubscribe_of_unknown_queue_is_a_noop():
    # Subscribing, then unsubscribing a queue that was never registered
    # (e.g. a stale handle from a torn-down connection) must not raise.
    q = generation_events.bus.subscribe()
    generation_events.bus.unsubscribe(q)
    generation_events.bus.unsubscribe(q)  # second call: must not raise
    assert generation_events.bus.subscriber_count == 0


def test_publish_fans_out_to_every_subscriber():
    q1 = generation_events.bus.subscribe()
    q2 = generation_events.bus.subscribe()
    generation_events.bus.publish("generation", {"id": "abc", "status": "completed"})
    assert _drain(q1) == {"topic": "generation", "data": {"id": "abc", "status": "completed"}}
    assert _drain(q2) == {"topic": "generation", "data": {"id": "abc", "status": "completed"}}
    assert q1.empty()
    assert q2.empty()


def test_publish_does_not_block_when_subscriber_queue_is_full():
    # The default queue is 256 entries. The 257th publish must NOT
    # raise or block (which would freeze the publisher coroutine in
    # the history route); the oldest entry is dropped instead.
    #
    # Note on asyncio.Queue + synchronous publish(): Queue.put_nowait
    # works fine outside an event loop (it just signals the loop's
    # wakeup no-op), and the drop-oldest path in the bus is correct
    # under both sync and async contexts. We run sync here for speed.
    q = generation_events.bus.subscribe()
    for i in range(256):
        generation_events.bus.publish("generation", {"i": i})
    # 257th publish -- must complete without raising and the queue
    # must stay bounded.
    generation_events.bus.publish("generation", {"i": 256})
    assert q.qsize() <= 256, f"queue grew beyond cap: {q.qsize()}"
    # Under FIFO drop-oldest, the *newest* message must survive.
    # Drain and check the last element is the i=256 one.
    last = None
    while not q.empty():
        last = q.get_nowait()
    assert last == {"topic": "generation", "data": {"i": 256}}, (
        f"newest message was dropped: got {last!r}"
    )
