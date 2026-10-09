from scripts.phyediting_auxiliary_common_trial import common_auxiliary


def test_trimming_auxiliary_preserves_absolute_motion_time_and_primary_input():
    sample={'image_size':[100,80],'boxes':{'frames':[1,3,5,7,9], 'xyxy':[[0,10,20,30]]+[[10,10,20,30]]*4,'times':[0,.1,.2,.3,.4]},
            'motion':{'t0':[.05,.05]},'window':{'max_frames':5}}
    result,n=common_auxiliary(sample)
    assert n==4 and result['boxes']['frames']==[3,5,7,9]
    assert abs(result['motion']['t0'][0]-.15)<1e-10
    assert sample['motion']['t0']==[.05,.05] and len(sample['boxes']['frames'])==5
    sample['window']['edges_visible']={'3':[True,True,False,True]}
    assert common_auxiliary(sample)==(None,3)
