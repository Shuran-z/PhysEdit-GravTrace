from scripts.phyediting_subpixel_eval import reindex


def test_subsample_preserves_real_timestamps_and_never_interpolates():
    tr = {"object": {"frames": [22, 23, 24, 25, 27, 28], "xyxy": [[f, 0, f + 1, 2] for f in [22, 23, 24, 25, 27, 28]]}}
    result = reindex(tr, 30, 15)
    assert result["tracks"]["object"]["frames"] == [0, 1, 2]
    assert [b[0] for b in result["tracks"]["object"]["xyxy"]] == [23, 25, 27]
    assert tr["object"]["frames"][0] == 22
