import pytest
from experiments.astra_world_demo_v1.proxy_geometry import containment_scale, containment_center


def test_oversized_nested_object_is_shrunk_not_container_expanded():
    factor = containment_scale([-.4032, -.4032, -.1092], [.4032, .4032, .0756],
                               [-.3045, -.3045, -.1155], [.3045, .3045, .1155])
    assert 0 < factor < 1
    assert factor * .8064 <= .609 - .01 + 1e-9


def test_small_object_keeps_dimensions_and_offset_is_clamped():
    assert containment_scale([0, 0, 0], [.1, .1, .1], [0, 0, 0], [1, 1, 1]) == 1
    assert containment_center([0, 0, 0], [.1, .1, .1], [0, 0, 0], [1, 1, 1], offset_x=100)[0] == .945


@pytest.mark.parametrize('maximum', [[0, 1, 1], [float('nan'), 1, 1]])
def test_invalid_geometry_rejected(maximum):
    with pytest.raises(ValueError):
        containment_scale([0, 0, 0], maximum, [0, 0, 0], [1, 1, 1])
