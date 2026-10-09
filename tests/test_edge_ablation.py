from scripts.phyediting_edge_ablation import mask_edges


def test_axis_ablation_preserves_frames_boxes_and_hidden_edges():
    boxes = {'frames': [1, 3, 5], 'xyxy': [[1,2,9,10]]*3}
    s = {'boxes': boxes, 'window': {'edges_visible': {'3': [True, False, True, True]}}}
    mask_edges(s, 'vertical')
    assert s['boxes'] is boxes
    assert s['window']['edges_visible'] == {'1': [False,True,False,True],
                                          '3': [False,False,False,True],
                                          '5': [False,True,False,True]}
    assert s['window']['contact_check'] is False
