from nflprops.simulation.props import prop_confidence_tier


def test_unknown_provider_prop_type_is_not_publishable():
    assert prop_confidence_tier("anytime_td_2q") is None


def test_unknown_quarter_yardage_market_is_not_publishable():
    assert prop_confidence_tier("passing_yards_2q") is None


def test_known_core_prop_remains_publishable():
    assert prop_confidence_tier("passing_yards") == 1


def test_known_tier_two_prop_remains_publishable():
    assert prop_confidence_tier("anytime_td") == 2
