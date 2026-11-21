import pytest

from core.page_hinkley import PageHinkley


def test_page_hinkley_detects_sustained_error_increase():
    monitor = PageHinkley(delta=0.03, threshold=0.25)
    for _ in range(20):
        assert not monitor.update(0.0)
    alarms = [monitor.update(1.0) for _ in range(10)]
    assert any(alarms)
    assert monitor.statistic > monitor.threshold


def test_page_hinkley_reset_clears_state():
    monitor = PageHinkley(delta=0.01, threshold=0.1)
    for _ in range(5):
        monitor.update(1.0)
    monitor.reset()
    assert monitor.statistic == 0
    assert monitor.count == 0
    assert monitor.mean == 0


def test_page_hinkley_rejects_invalid_observation():
    with pytest.raises(ValueError):
        PageHinkley().update(1.2)
