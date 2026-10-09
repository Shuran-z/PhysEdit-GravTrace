from scripts.phyediting_bounded_depth import lift_job
from scripts.phyediting_unknown_common import remove_velocity
from tests.test_synthetic import sample


def test_missing_depth_cannot_silently_drop_a_fit_frame():
    s=remove_velocity(sample('projectile',{'v0':[999,999,999],'gravity_dir':[0,0,-1]},4))
    s['boxes']={'frames':[0,1,2,3],'xyxy':[[310,170,330,190]]*4};s['window']['start_frame']=0
    assert lift_job(s,{'frames':[0,1,2],'depth':[4,4,4]})==(None,'missing_depth')
    data,error=lift_job(s,{'frames':[0,1,2,3],'depth':[4,4,4,4]})
    assert error is None and len(data[0])==4 and data[1].shape==(4,3)
