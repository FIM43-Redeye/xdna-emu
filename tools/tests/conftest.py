"""Registers the markers used by the tools/tests suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "aiecc: runs the real aiecc compiler (minutes); deselect with -m 'not aiecc'")
