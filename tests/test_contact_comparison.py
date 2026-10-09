from scripts.phyediting_contact_comparison import summary


def test_summary_counts_unobservable_failures_in_denominator():
    rows = [{'error_pct': 5., 'status': 'ok'}, {'error_pct': None, 'status': 'no_track'}]
    s = summary(rows)
    assert s['coverage'] == .5 and s['mean_pct'] == 5.
    assert s['attempted'] == 2 and s['within10_all'] == 1
    assert summary([])['mean_pct'] is None
