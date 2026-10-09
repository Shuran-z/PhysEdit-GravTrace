import pytest
from scripts.phyediting_vggt_common_eval import merge_records


def test_common_merge_rejects_duplicates_and_wrong_order():
    samples=[{'id':'x','boxes':{'frames':[4,7]}}]
    record={'id':'x','model':'vggt','frames':[4,7],'depth':[1.,2.]}
    assert merge_records(samples,[record],[])['x']==record
    with pytest.raises(ValueError):merge_records(samples,[record],[record])
    with pytest.raises(ValueError):merge_records(samples,[],[])
    with pytest.raises(ValueError):merge_records(samples,[dict(record,frames=[7,4])],[])
