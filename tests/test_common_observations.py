from scripts.phyediting_common_observations import restrict


def test_common_frames_keep_initial_state_and_correct_elapsed_time():
    s = {"fps": 30, "image_size": [100, 100], "motion": {"v0": [1, 0, 0], "t0": [0, 0]},
         "boxes": {"frames": [10, 11, 12, 13, 14, 15], "xyxy": [[5, 5, 20, 20]] * 6},
         "window": {"start_frame": 10, "max_frames": 6, "edges_visible": {"10": [False] * 4}}}
    r = restrict(s)
    assert r["boxes"]["frames"] == [11, 12, 13, 14, 15]
    assert r["motion"]["t0"] == [1 / 30, 1 / 30]
    assert r["motion"]["v0"] == s["motion"]["v0"]
    assert s["motion"]["t0"] == [0, 0]
    s["boxes"]["xyxy"] = [[0, 5, 20, 20]] * 6
    assert restrict(s) is None
