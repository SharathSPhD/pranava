"""The instruction template/extraction layer is pure — test it on host (no GPU)."""
from pranava.alm.instruct import INSTRUCTIONS, tasks_for


def test_sanskrit_clip_yields_transcribe_language_and_filled_karakas():
    karaka = [["bAlAH", "karwA"], ["vidyayA", "karaNam"], ["pustakam", "karma"], ["KAdeyuH", "kriyA"]]
    ex = tasks_for(wav="u0.wav", text="bAlAH vidyayA pustakam KAdeyuH", karaka=karaka,
                   lang="sa", split="train")
    by_task = {e.task: e.response for e in ex}
    assert by_task["transcribe"] == "bAlAH vidyayA pustakam KAdeyuH"
    assert by_task["language"] == "Sanskrit"
    assert by_task["karta"] == "bAlAH"       # agent
    assert by_task["karana"] == "vidyayA"    # instrument
    assert by_task["karma"] == "pustakam"    # object
    assert by_task["kriya"] == "KAdeyuH"     # action
    # instructions are natural English, not opaque task ids
    assert all(e.instruction in INSTRUCTIONS.values() for e in ex)


def test_absent_karaka_role_is_not_invented():
    # only kartā + kriyā present → no karma/karaṇa examples emitted (never fabricated)
    ex = tasks_for(wav="u1.wav", text="devaH gacCati", karaka=[["devaH", "karwA"], ["gacCati", "kriyA"]],
                   lang="sa", split="val")
    tasks = {e.task for e in ex}
    assert "karma" not in tasks and "karana" not in tasks
    assert tasks == {"transcribe", "language", "karta", "kriya"}


def test_english_clip_strips_lang_tag_and_has_no_karaka_tasks():
    ex = tasks_for(wav="en0.wav", text="[en] hello there", karaka=[], lang="en", split="train")
    by_task = {e.task: e.response for e in ex}
    assert by_task["transcribe"] == "hello there"   # [en] tag stripped
    assert by_task["language"] == "English"
    assert set(by_task) == {"transcribe", "language"}
