"""
Offline tests for flux experiment variants.

Verifies:
1. All flux classes import correctly via meas.*
2. ResSpec flux parameters are set up correctly
3. RabiFluxProgram body uses delay (not delay_auto)
4. HistogramFluxProgram body uses delay (not delay_auto)
5. Default flux parameters use sweet_spot_ac

Usage: python test_flux_experiments.py
"""
import os
import sys
import inspect
import ast

def test_imports():
    """Verify all flux classes are importable via the meas namespace."""
    import slab_qick_calib.experiments as meas

    flux_classes = [
        'RabiFluxProgram',
        'RabiFluxExperiment',
        'HistogramFluxProgram',
        'HistogramFluxExperiment',
    ]
    for cls_name in flux_classes:
        assert hasattr(meas, cls_name), f"Missing export: {cls_name}"
        print(f"  {cls_name}: OK")

    # ResSpec should also have flux params (same class, not a copy)
    assert hasattr(meas, 'ResSpec'), "Missing ResSpec"
    print("  ResSpec: OK")
    print("All imports passed!")


def test_no_delay_auto_in_flux_programs():
    """
    Verify that flux program _body methods use delay() not delay_auto().
    This is critical: delay_auto waits for all pulses (including the long flux
    pulse) to finish, which would break the concurrent flux pattern.
    """
    scripts_dir = os.path.dirname(__file__)
    base_dir = os.path.join(scripts_dir, '..', 'experiments', 'single_qubit')

    files_to_check = {
        'rabi_flux.py': 'RabiFluxProgram',
        'single_shot_flux.py': 'HistogramFluxProgram',
    }

    for filename, class_name in files_to_check.items():
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '_body':
                        # Check all calls in _body for delay_auto
                        for subnode in ast.walk(item):
                            if isinstance(subnode, ast.Attribute):
                                if subnode.attr == 'delay_auto':
                                    raise AssertionError(
                                        f"{class_name}._body in {filename} "
                                        f"contains delay_auto at line {subnode.lineno}! "
                                        f"Must use delay() instead."
                                    )
                        print(f"  {class_name}._body: no delay_auto found - OK")

    print("No delay_auto in flux program bodies - passed!")


def test_flux_pulse_in_initialize():
    """Verify flux programs declare the flux generator and create flux pulse in _initialize."""
    scripts_dir = os.path.dirname(__file__)
    base_dir = os.path.join(scripts_dir, '..', 'experiments', 'single_qubit')

    files_to_check = {
        'rabi_flux.py': 'RabiFluxProgram',
        'single_shot_flux.py': 'HistogramFluxProgram',
    }

    for filename, class_name in files_to_check.items():
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'r') as f:
            source = f.read()

        # Check for key patterns in _initialize
        assert 'declare_gen' in source, f"{filename}: missing declare_gen"
        assert 'flux_pulse' in source, f"{filename}: missing flux_pulse"
        assert 'mixer_freq=0' in source, f"{filename}: missing mixer_freq=0 for DC flux"
        print(f"  {filename}: flux setup in _initialize - OK")

    print("Flux pulse setup verified!")


def test_flux_in_body():
    """Verify flux programs play the flux pulse at the start of _body."""
    scripts_dir = os.path.dirname(__file__)
    base_dir = os.path.join(scripts_dir, '..', 'experiments', 'single_qubit')

    files_to_check = {
        'rabi_flux.py': 'RabiFluxProgram',
        'single_shot_flux.py': 'HistogramFluxProgram',
    }

    for filename, class_name in files_to_check.items():
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '_body':
                        # Get the source lines of _body
                        body_lines = source.split('\n')[item.lineno-1:item.end_lineno]
                        body_text = '\n'.join(body_lines)

                        assert 'flux_pulse' in body_text, \
                            f"{class_name}._body missing flux_pulse playback"
                        assert 'flux_settle' in body_text, \
                            f"{class_name}._body missing flux_settle delay"

                        # Verify flux pulse comes before qubit pulse
                        flux_idx = body_text.index('flux_pulse')
                        qubit_idx = body_text.index('qubit_ch') if 'qubit_ch' in body_text else body_text.index('pi_ge')
                        assert flux_idx < qubit_idx, \
                            f"{class_name}._body: flux_pulse must come before qubit pulses"

                        print(f"  {class_name}._body: flux pulse played first - OK")

    print("Flux pulse ordering verified!")


def test_default_params():
    """Verify flux experiment classes default to sweet_spot_ac."""
    from slab_qick_calib.experiments.single_qubit.rabi_flux import RabiFluxExperiment
    from slab_qick_calib.experiments.single_qubit.single_shot_flux import HistogramFluxExperiment

    # Check source code for sweet_spot_ac reference in defaults
    for cls in [RabiFluxExperiment, HistogramFluxExperiment]:
        source = inspect.getsource(cls.__init__)
        assert 'sweet_spot_ac' in source, \
            f"{cls.__name__}.__init__ missing sweet_spot_ac default for flux_gain"
        assert 'flux_settle' in source, \
            f"{cls.__name__}.__init__ missing flux_settle default"
        print(f"  {cls.__name__}: sweet_spot_ac default - OK")

    print("Default parameters verified!")


def test_resspec_flux_params():
    """Verify ResSpec has flux parameters in its defaults."""
    from slab_qick_calib.experiments.single_qubit.resonator_spectroscopy import ResSpec

    source = inspect.getsource(ResSpec.__init__)
    assert 'flux' in source, "ResSpec.__init__ missing flux parameter"
    assert 'sweet_spot_ac' in source, "ResSpec.__init__ missing sweet_spot_ac default"
    assert 'flux_settle' in source, "ResSpec.__init__ missing flux_settle default"
    print("  ResSpec: flux params in defaults - OK")

    # Also check the program
    from slab_qick_calib.experiments.single_qubit.resonator_spectroscopy import ResSpecProgram

    source = inspect.getsource(ResSpecProgram._body)
    assert 'flux_pulse' in source, "ResSpecProgram._body missing flux_pulse"
    assert 'flux_settle' in source, "ResSpecProgram._body missing flux_settle"
    print("  ResSpecProgram._body: flux support - OK")

    # Check _initialize
    source = inspect.getsource(ResSpecProgram._initialize)
    assert 'declare_gen' in source, "ResSpecProgram._initialize missing declare_gen for flux"
    assert 'flux_pulse' in source, "ResSpecProgram._initialize missing flux_pulse creation"
    print("  ResSpecProgram._initialize: flux setup - OK")

    print("ResSpec flux parameters verified!")


if __name__ == "__main__":
    print("=== 1. Import test ===")
    test_imports()

    print("\n=== 2. No delay_auto in flux programs ===")
    test_no_delay_auto_in_flux_programs()

    print("\n=== 3. Flux pulse in _initialize ===")
    test_flux_pulse_in_initialize()

    print("\n=== 4. Flux pulse ordering in _body ===")
    test_flux_in_body()

    print("\n=== 5. Default parameters ===")
    test_default_params()

    print("\n=== 6. ResSpec flux parameters ===")
    test_resspec_flux_params()

    print("\n=============================")
    print("All tests passed!")
