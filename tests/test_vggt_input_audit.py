from scripts.phyediting_vggt_common_jobs import same_input


def test_reuse_requires_identical_order_boxes_and_camera():
    w={'frames':[1,3,5],'boxes':[[1,2,3,4]]*3};K=[[1,0,0],[0,1,0],[0,0,1]]
    assert same_input(w,w,K,K)
    assert not same_input({**w,'frames':[5,3,1]},w,K,K)
    assert not same_input({**w,'boxes':[[2,2,3,4]]*3},w,K,K)
    assert not same_input(w,w,[[2,0,0],[0,1,0],[0,0,1]],K)
