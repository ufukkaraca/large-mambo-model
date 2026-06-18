"""Session state (mambo.session.v1) save/load/list round-trip — the F1 gate
(`the session contract` §4): save → restore → byte-identical
`session_context` contract keys; no contract schema touched."""

import json

from mambo_lab import oracle, session


def test_slugify():
    assert session.slugify("Midnight Drive") == "midnight-drive"
    assert session.slugify("  A/B Test!! ") == "a-b-test"
    assert session.slugify("").startswith("session-")


def test_save_load_roundtrip(tmp_path):
    sess = session.create("Midnight Drive", root=tmp_path)
    assert sess["slug"] == "midnight-drive"
    assert (tmp_path / "midnight-drive" / "session.json").exists()

    loaded = session.load("midnight-drive", root=tmp_path)
    assert loaded["name"] == "Midnight Drive"
    assert loaded["schema"] == "mambo.session.v1"
    assert loaded["phase"] == "idle"
    assert loaded["takes"] == []


def test_context_keys_are_byte_identical_to_template(tmp_path):
    """The contract `session_context` block a fresh session yields must equal the
    cold-start template byte-for-byte — nothing downstream may notice the swap."""
    sess = session.create("Demo", root=tmp_path)
    loaded = session.load("demo", root=tmp_path)
    projected = session.session_context(loaded)
    template = oracle.default_session_context()
    assert json.dumps(projected, sort_keys=True) == json.dumps(template, sort_keys=True)


def test_seed_overrides_contract_keys(tmp_path):
    seed = {"project_tempo_bpm": 96, "project_key": "A minor"}
    sess = session.create("Seeded", root=tmp_path, seed=seed)
    loaded = session.load("seeded", root=tmp_path)
    assert loaded["project_tempo_bpm"] == 96
    assert loaded["project_key"] == "A minor"
    # un-seeded keys still come from the template
    assert loaded["transport"] == "stopped"


def test_notebook_persists(tmp_path):
    session.create("Notes", root=tmp_path)
    session.write_notebook("notes", "walking down the avenue\n", root=tmp_path)
    assert session.read_notebook("notes", root=tmp_path) == "walking down the avenue\n"
    # missing notebook reads as empty, never raises
    assert session.read_notebook("nope", root=tmp_path) == ""


def test_list_sessions_newest_first(tmp_path):
    session.create("First", root=tmp_path)
    session.create("Second", root=tmp_path)
    listed = session.list_sessions(root=tmp_path)
    slugs = [s["slug"] for s in listed]
    assert set(slugs) == {"first", "second"}
    # newest-updated first (Second created last)
    assert slugs[0] == "second"
    assert all("name" in s and "takes" in s for s in listed)


def test_current_pointer_and_session_context(tmp_path):
    # no current session → cold-start template
    assert session.load_session_context(root=tmp_path) == oracle.default_session_context()
    session.create("Active", root=tmp_path, seed={"project_tempo_bpm": 140})
    assert session.current_slug(root=tmp_path) == "active"
    ctx = session.load_session_context(root=tmp_path)
    assert ctx["project_tempo_bpm"] == 140
    # ctx carries exactly the contract keys, no session-only fields leak in
    assert set(ctx) == {"daw", "selected_track", "tracks", "project_tempo_bpm",
                        "project_key", "transport"}


def test_switch_and_set_phase(tmp_path):
    session.create("Alpha", root=tmp_path)
    session.create("Beta", root=tmp_path)
    assert session.current_slug(root=tmp_path) == "beta"  # last created is current
    session.switch("alpha", root=tmp_path)
    assert session.current_slug(root=tmp_path) == "alpha"
    session.set_phase("alpha", "tracking_vocals", root=tmp_path)
    assert session.load("alpha", root=tmp_path)["phase"] == "tracking_vocals"


def test_append_take(tmp_path):
    session.create("Takes", root=tmp_path)
    session.append_take("takes", {"id": "vox1-001", "track": "Vox 1", "kept": False}, root=tmp_path)
    session.append_take("takes", {"id": "vox1-002", "track": "Vox 1", "kept": True}, root=tmp_path)
    loaded = session.load("takes", root=tmp_path)
    assert [t["id"] for t in loaded["takes"]] == ["vox1-001", "vox1-002"]  # newest last
    assert loaded["takes"][-1]["kept"] is True


def test_rename_keeps_slug(tmp_path):
    session.create("Old Name", root=tmp_path)
    session.rename("old-name", "New Name", root=tmp_path)
    loaded = session.load("old-name", root=tmp_path)  # dir/slug stable
    assert loaded["name"] == "New Name"
    assert loaded["slug"] == "old-name"


def test_take_log(tmp_path):
    s = session.create("demo", root=tmp_path); slug = s["slug"]
    session.append_take(slug, {"track": {"by": "Bass"}}, root=tmp_path)
    session.append_take(slug, {"track": {"by": "Bass"}, "label": "loose"}, root=tmp_path)
    takes = session.load(slug, root=tmp_path)["takes"]
    assert [t["id"] for t in takes] == [1, 2] and takes[0]["kept"] is False
    kept = session.mark_take_kept(slug, label="the one", root=tmp_path)
    assert kept["id"] == 2 and kept["kept"] is True and kept["label"] == "the one"
    assert session.find_take(slug, 1, root=tmp_path)["id"] == 1
    assert session.find_take(slug, "keeper", root=tmp_path)["id"] == 2
    assert session.find_take(slug, 9, root=tmp_path) is None
    dropped = session.scratch_take(slug, root=tmp_path)
    assert dropped["id"] == 2 and len(session.load(slug, root=tmp_path)["takes"]) == 1


def test_list_sessions_memory_summary(tmp_path):
    # the Projects UI needs a per-project memory summary in the list
    s = session.create("Mix A", root=tmp_path); slug = s["slug"]
    session.append_take(slug, {"track": {"by": "Vox"}}, root=tmp_path)
    session.write_notebook(slug, "line one\n\nline two\n", root=tmp_path)
    row = next(r for r in session.list_sessions(root=tmp_path) if r["slug"] == slug)
    assert row["takes"] == 1 and row["notebook_lines"] == 2 and row["has_reaper"] is False


def test_reaper_project_path_creates_valid_rpp(tmp_path):
    s = session.create("Beat", root=tmp_path); slug = s["slug"]
    assert not session.reaper_project_path(slug, root=tmp_path, create=False).exists()
    rpp = session.reaper_project_path(slug, root=tmp_path, create=True)
    assert rpp.exists() and rpp.suffix == ".rpp"
    assert rpp.read_text().startswith("<REAPER_PROJECT") and "TEMPO" in rpp.read_text()
    row = next(r for r in session.list_sessions(root=tmp_path) if r["slug"] == slug)
    assert row["has_reaper"] is True  # now the project has a REAPER document


def test_take_by_id_for_ui(tmp_path):
    # the UI keeps/scratches a SPECIFIC take by id (not just the latest)
    s = session.create("ui", root=tmp_path); slug = s["slug"]
    for _ in range(3):
        session.append_take(slug, {"track": {"by": "Vox"}}, root=tmp_path)
    assert [t["id"] for t in session.read_takes(slug, root=tmp_path)] == [1, 2, 3]
    # star take 1 (the earliest) + relabel — not the latest, which mark_take_kept can't do
    t1 = session.set_take(slug, 1, kept=True, label="warm one", root=tmp_path)
    assert t1["kept"] is True and t1["label"] == "warm one"
    assert session.find_take(slug, "keeper", root=tmp_path)["id"] == 1
    # un-keep is also reachable
    assert session.set_take(slug, 1, kept=False, root=tmp_path)["kept"] is False
    # scratch the middle take by id → ids of the rest stay stable (no renumber)
    assert session.remove_take(slug, 2, root=tmp_path)["id"] == 2
    assert [t["id"] for t in session.read_takes(slug, root=tmp_path)] == [1, 3]
    # unknown id → None, no crash
    assert session.set_take(slug, 99, kept=True, root=tmp_path) is None
    assert session.remove_take(slug, 99, root=tmp_path) is None
