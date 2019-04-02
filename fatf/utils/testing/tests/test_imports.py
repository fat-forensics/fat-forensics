"""
Tests helper functions for unit-testing module imports.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import importlib
import sys

import pytest

import fatf.utils.testing.imports as futi


def test_module_import_tester():
    """
    Tests importing modules in various contexts.
    """
    # When the module is installed and we test assuming that it is installed.
    # Import should succeed.
    import pytest
    importlib.reload(pytest)
    with futi.module_import_tester('pytest', when_missing=True):
        # Import should fail.
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(pytest)
        assert 'module pytest not in sys.modules' in str(excinfo.value)
        with pytest.raises(ImportError) as excinfo:
            import pytest
        assert 'No module named \'pytest\'' in str(excinfo.value)
    # import should succeed.
    import pytest
    importlib.reload(pytest)

    # When the module is installed and we pretend that it is NOT installed.
    # Import should succeed.
    import pytest
    importlib.reload(pytest)
    with futi.module_import_tester('existing_module', when_missing=False):
        # Import should succeed.
        importlib.reload(sys)
        import pytest
    # Import should succeed.
    import pytest
    importlib.reload(pytest)

    missing_mod = 'gibberish_module_42'
    missing_mod_exception = 'No module named \'{}\''.format(missing_mod)
    # When the module is NOT installed and we test assuming that it is there.
    # Import should fail.
    with pytest.raises(ImportError) as excinfo:
        import gibberish_module_42
    assert missing_mod_exception in str(excinfo.value)
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(missing_mod)
    assert missing_mod_exception in str(excinfo.value)
    with futi.module_import_tester(missing_mod, when_missing=True):
        # Import should fail.
        with pytest.raises(ImportError) as excinfo:
            import gibberish_module_42
        assert missing_mod_exception in str(excinfo.value)
        with pytest.raises(ImportError) as excinfo:
            importlib.import_module(missing_mod)
        assert missing_mod_exception in str(excinfo.value)
    # Import should fail.
    with pytest.raises(ImportError) as excinfo:
        import gibberish_module_42
    assert missing_mod_exception in str(excinfo.value)
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(missing_mod)
    assert missing_mod_exception in str(excinfo.value)

    # When the module is NOT installed and we assume that it is NOT installed.
    # Import should fail.
    with pytest.raises(ImportError) as excinfo:
        import gibberish_module_42
    assert missing_mod_exception in str(excinfo.value)
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(missing_mod)
    assert missing_mod_exception in str(excinfo.value)
    with futi.module_import_tester(missing_mod, when_missing=False):
        # Import should succeed.
        import gibberish_module_42
    with futi.module_import_tester(missing_mod, when_missing=False):
        # Import should succeed.
        importlib.import_module(missing_mod)
    # Import should fail.
    with pytest.raises(ImportError) as excinfo:
        import gibberish_module_42
    assert missing_mod_exception in str(excinfo.value)
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(missing_mod)
    assert missing_mod_exception in str(excinfo.value)
