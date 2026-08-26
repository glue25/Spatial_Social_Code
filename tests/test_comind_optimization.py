from spatial_social_code.comind_optimization import normalize_prediction, sanitize_bbox


def test_auto_keeps_valid_ambiguous_xywh():
    box, action = sanitize_bbox([100, 100, 200, 300])
    assert box == [100, 100, 200, 300]
    assert action.endswith("kept_xywh")


def test_auto_converts_xyxy_only_when_xywh_impossible():
    box, action = sanitize_bbox([547, 720, 617, 820])
    assert box == [547, 720, 70, 100]
    assert "xyxy_to_xywh" in action


def test_normalized_and_clipped_coordinates():
    box, action = sanitize_bbox([0.5, 0.5, 0.75, 1.0], convention="xyxy")
    assert box == [704, 704, 352, 704]
    assert action.startswith("scaled_normalized")


def test_malformed_or_degenerate_box_is_rejected():
    assert sanitize_bbox([1, 2, 3])[0] is None
    assert sanitize_bbox([100, 100, -4, 20])[0] is None


def test_explicit_conventions_and_prediction_copy():
    xyxy, _ = sanitize_bbox({"x1": 10, "y1": 20, "x2": 30, "y2": 50})
    assert xyxy == [10, 20, 20, 30]
    original = {"bbox": [547, 720, 617, 820], "cue_type": ["verbal"]}
    normalized, actions = normalize_prediction("scoia", original)
    assert original["bbox"] == [547, 720, 617, 820]
    assert normalized["bbox"] == [547, 720, 70, 100]
    assert actions["native_pixels:xyxy_to_xywh"] == 1
