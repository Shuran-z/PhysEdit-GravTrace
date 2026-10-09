import numpy as np
from scripts.phyediting_motion_refine import refine_box


def test_motion_component_refines_box_and_falls_back_without_change():
    bg=np.zeros((80,100,3),np.uint8);image=bg.copy();image[20:40,30:50]=100
    result,reason=refine_box(image,bg,[28,18,53,42])
    assert result==[30,20,50,40] and reason=='refined'
    result,reason=refine_box(bg,bg,[28,18,53,42])
    assert result==[28,18,53,42] and reason=='no_component'
    image[5:70,5:90]=100
    assert refine_box(image,bg,[28,18,53,42])[1]=='no_component'
