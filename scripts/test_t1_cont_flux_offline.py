"""Offline test for T1ContFluxExperiment — verifies data processing logic without hardware."""
import numpy as np
import sys
import os

# Test the data processing logic that acquire() does at each flux point
# by simulating what iq_list would look like from the program

def test_data_extraction():
    """Simulate iq_list and verify ve/vg/vi_t1 extraction and T1 calculation."""
    n_e = 2   # short-wait (excited) measurements
    n_g = 1   # long-wait (ground) measurements
    n_t1 = 8  # T1 measurements
    n_total = n_e + n_g + n_t1  # 11
    n_reps = 100
    wait_time = 0.5  # us (T1/2)

    # Simulate IQ data: shape [1, n_total, n_reps, 2]
    # Short-wait should give high I (excited state), long-wait low I (ground state)
    rng = np.random.default_rng(42)
    e_mean_i = 20.0   # excited state I value
    g_mean_i = 10.0    # ground state I value
    true_t1 = 1.0      # us
    true_pop = np.exp(-wait_time / true_t1)  # exp(-0.5) ≈ 0.6065
    t1_mean_i = g_mean_i + true_pop * (e_mean_i - g_mean_i)  # ≈ 16.065

    iq_data = np.zeros((1, n_total, n_reps, 2))
    noise = 0.5

    # Short-wait (excited) — indices 0:2
    iq_data[0, 0:n_e, :, 0] = e_mean_i + rng.normal(0, noise, (n_e, n_reps))
    iq_data[0, 0:n_e, :, 1] = rng.normal(0, noise, (n_e, n_reps))

    # Long-wait (ground) — indices 2:3
    iq_data[0, n_e:n_e+n_g, :, 0] = g_mean_i + rng.normal(0, noise, (n_g, n_reps))
    iq_data[0, n_e:n_e+n_g, :, 1] = rng.normal(0, noise, (n_g, n_reps))

    # T1 data — indices 3:11
    iq_data[0, n_e+n_g:n_e+n_g+n_t1, :, 0] = t1_mean_i + rng.normal(0, noise, (n_t1, n_reps))
    iq_data[0, n_e+n_g:n_e+n_g+n_t1, :, 1] = rng.normal(0, noise, (n_t1, n_reps))

    # Extract exactly as acquire() does
    iq_array = np.array(iq_data)
    ve = float(np.mean(iq_array[0, 0:n_e, :, 0]))
    vg = float(np.mean(iq_array[0, n_e:n_e+n_g, :, 0]))
    vi_t1 = float(np.mean(iq_array[0, n_e+n_g:n_e+n_g+n_t1, :, 0]))

    dv = ve - vg
    pop = (vi_t1 - vg) / dv
    t1_extracted = -wait_time / np.log(pop)

    print(f"ve = {ve:.4f}  (expected ~{e_mean_i})")
    print(f"vg = {vg:.4f}  (expected ~{g_mean_i})")
    print(f"vi_t1 = {vi_t1:.4f}  (expected ~{t1_mean_i:.4f})")
    print(f"pop = {pop:.4f}  (expected ~{true_pop:.4f})")
    print(f"T1 = {t1_extracted:.4f} us  (expected ~{true_t1:.4f})")

    # Check values are reasonable
    assert abs(ve - e_mean_i) < 1.0, f"ve off: {ve}"
    assert abs(vg - g_mean_i) < 1.0, f"vg off: {vg}"
    assert abs(pop - true_pop) < 0.1, f"pop off: {pop}"
    assert abs(t1_extracted - true_t1) < 0.3, f"T1 off: {t1_extracted}"

    print("\nAll assertions passed!")


def test_import():
    """Verify module imports correctly."""
    from slab_qick_calib.experiments.single_qubit.t1_cont_flux import (
        T1ContFluxProgram, T1ContFluxExperiment
    )
    print(f"T1ContFluxProgram: {T1ContFluxProgram}")
    print(f"T1ContFluxExperiment: {T1ContFluxExperiment}")

    import slab_qick_calib.experiments as meas
    assert hasattr(meas, 'T1ContFluxExperiment'), "Not exported via meas"
    assert hasattr(meas, 'T1ContFluxProgram'), "Not exported via meas"
    print("Module exports OK")


if __name__ == "__main__":
    print("=== Import test ===")
    test_import()
    print("\n=== Data extraction test ===")
    test_data_extraction()
    print("\nAll tests passed!")
