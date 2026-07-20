"""Errored downloads must not be reported as still downloading.

A failed download intentionally stays in the TaskManager with
``status="error"`` so ``/tasks/active`` can surface the error and retry
UI — but ``/models/status`` derives its ``downloading`` flag from the
same list. Without a status filter, one failed download shows the model
as "downloading" forever and masks its real cache state until the app
restarts (issue #925, symptom reports like #181).
"""

from backend.utils.tasks import TaskManager


def test_errored_download_is_not_pending():
    tm = TaskManager()
    tm.start_download("whisper-turbo")
    assert [t.model_name for t in tm.get_pending_downloads()] == ["whisper-turbo"]

    tm.error_download("whisper-turbo", "boom")

    assert tm.get_pending_downloads() == []
    # Still visible to /tasks/active for the error/retry UI.
    active = tm.get_active_downloads()
    assert [t.model_name for t in active] == ["whisper-turbo"]
    assert active[0].status == "error"
    assert active[0].error == "boom"


def test_retry_after_error_is_pending_again():
    tm = TaskManager()
    tm.start_download("qwen3-4b")
    tm.error_download("qwen3-4b", "boom")
    tm.start_download("qwen3-4b")
    assert [t.model_name for t in tm.get_pending_downloads()] == ["qwen3-4b"]


def test_completed_download_is_removed_everywhere():
    tm = TaskManager()
    tm.start_download("whisper-turbo")
    tm.complete_download("whisper-turbo")
    assert tm.get_pending_downloads() == []
    assert tm.get_active_downloads() == []


def test_cancel_dismisses_errored_download():
    tm = TaskManager()
    tm.start_download("whisper-turbo")
    tm.error_download("whisper-turbo", "boom")
    assert tm.cancel_download("whisper-turbo") is True
    assert tm.get_active_downloads() == []
    assert tm.get_pending_downloads() == []
