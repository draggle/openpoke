"""Clear resets everything the next conversation could see, including the rolling summary."""

from server.services.conversation import get_working_memory_log


def test_clear_history_also_clears_the_working_memory_summary(monkeypatch):
    from server.routes import chat

    wm = get_working_memory_log()
    wm.clear()
    wm.append_entry("user_message", "remind me to call mom tomorrow at 9pm")
    assert "call mom" in wm.render_transcript()

    chat.clear_history()

    assert "call mom" not in wm.render_transcript()
