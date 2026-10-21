from scripts.evaluate_sample import evaluate_sample


def test_the_sample_check_scores_our_library_by_default():
    result = evaluate_sample()
    assert result["detector"] == "rule_based"
    assert result["overall"]["f1"] == 1.0
    assert result["pulse_components_match"] is True


def test_the_sample_check_scores_any_registered_detector():
    result = evaluate_sample("team")
    assert result["detector"] == "team"
    assert 0.0 <= result["overall"]["f1"] <= 1.0
