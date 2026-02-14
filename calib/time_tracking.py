import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ..helpers import config
import slab_qick_calib.experiments as meas
from . import qubit_tuning

# Configure matplotlib
plt.rcParams["legend.handlelength"] = 0.5

# Default parameters
MAX_T1 = 500  # Maximum T1 time in microseconds
MAX_ERR = 1  # Maximum acceptable error in fits
MIN_R2 = 0.35  # Minimum acceptable R² value for fits
TOL = 0.3  # Tolerance for parameter convergence


def measure_params(
    qi,
    cfg_dict,
    update=True,
    readout=True,
    fast=False,
    check_fid=True,
    display=False,
    max_t1=MAX_T1,
):
    """
    Measure and return key parameters for a qubit.

    This function performs a series of measurements to characterize a qubit's properties,
    including coherence times, frequencies, and readout fidelity.

    Parameters
    ----------
    qi : int
        Qubit index
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    update : bool, optional
        Whether to update the configuration file with new parameters
    readout : bool, optional
        Whether to update readout frequency
    display : bool, optional
        Whether to display plots during measurements
    max_t1 : float, optional
        Maximum T1 time in microseconds

    Returns
    -------
    dict
        Dictionary containing measured qubit parameters
    """
    cfg_path = cfg_dict["cfg_file"]
    err_dict = {}

    if not fast:
        # Step 1: Resonator spectroscopy
        rspec = meas.ResSpec(
            cfg_dict, qi=qi, params={"span": "kappa"}, display=display, progress=False
        )
        if update:
            rspec.update(freq=readout, fast=True, verbose=False)

        if not rspec.status:
            # Handle failed resonator spectroscopy
            rspec.data["kappa"] = np.nan
            rspec.data["fit"] = [np.nan, np.nan, np.nan]
            rspec.display(debug=True)
            print("Resonator spectroscopy failed")
    if not fast or check_fid:
        # Step 2: Single shot measurement
        shot = meas.HistogramExperiment(
            cfg_dict, qi=qi, params={"shots": 10000}, display=display, progress=False
        )
        if update:
            shot.update(fast=True, verbose=False)
    if not fast:
        # Step 3: Amplitude Rabi
        amp_rabi = meas.RabiExperiment(
            cfg_dict,
            qi=qi,
            params={"start": 0.003},
            display=display,
            progress=False,
            style="fast",
        )
        if update and amp_rabi.status:
            config.update_qubit(
                cfg_path,
                ("pulses", "pi_ge", "gain"),
                amp_rabi.data["pi_length"],
                qi,
                verbose=False,
            )

        if not amp_rabi.status:
            amp_rabi.data["pi_length"] = np.nan
            print("Amp Rabi failed")

        err_dict["rabi_err"] = np.sqrt(amp_rabi.data["fit_err_avgi"][1][1])

    # Step 4: T2 Ramsey
    t2r = meas.T2Experiment(
        cfg_dict, qi=qi, display=display, progress=False, style="fast"
    )
    if t2r.status and update:
        # Update qubit frequency and T2r time
        config.update_qubit(cfg_path, "f_ge", t2r.data["new_freq"], qi, verbose=False)
        config.update_qubit(
            cfg_path,
            "T2r",
            t2r.data["best_fit"][3],
            qi,
            rng_vals=[1, max_t1],
            sig=2,
            verbose=False,
        )

    if not t2r.status:
        recenter(qi, cfg_dict, t2r, update=update, display=display, max_t1=max_t1)

    err_dict["t2r_err"] = np.sqrt(t2r.data["fit_err_avgi"][3][3])
    err_dict["fge_err"] = np.sqrt(t2r.data["fit_err_avgi"][1][1])

    # Step 5: T1 measurement
    # t1 = meas.T1Experiment(cfg_dict, qi=qi, display=display, progress=False, style='fast')
    t1 = meas.T1Experiment(
        cfg_dict, qi=qi, display=display, progress=False, params={"span": 300}
    )
    if update:
        t1.update(rng_vals=[1, max_t1], verbose=False)

    if not t1.status:
        t1.data["new_t1_i"] = np.nan
        t1.display(debug=True)
        print("T1 failed")

    err_dict["t1_err"] = np.sqrt(t1.data["fit_err_avgi"][2][2])

    if not fast:
        # Step 6: T2 Echo measurement
        t2e = meas.T2Experiment(
            cfg_dict,
            qi=qi,
            display=display,
            progress=False,
            params={"experiment_type": "echo"},
            style="fast",
        )
        if update and t2e.status:
            config.update_qubit(
                cfg_path,
                "T2e",
                t2e.data["best_fit"][3],
                qi,
                rng_vals=[1, max_t1],
                sig=2,
                verbose=False,
            )

        if not t2e.status:
            # Try to recover from failed T2E measurement
            print("Refitting")
            t2e.analyze(refit=True, verbose=True)
            if not t2e.status:
                t2e.data["best_fit"] = [np.nan, np.nan, np.nan, np.nan]
                t2e.display(debug=True)
                print("T2 Echo failed")

        err_dict["t2e_err"] = np.sqrt(t2e.data["fit_err_avgi"][3][3])

    if not fast:
        # Compile all measured parameters into a dictionary
        qubit_dict = {
            "t1": t1.data["new_t1_i"],
            "t2r": t2r.data["best_fit"][3],
            "t2e": t2e.data["best_fit"][3],
            "f_ge": t2r.data["new_freq"],
            "fidelity": shot.data["fids"][0],
            "phase": shot.data["angle"],
            "kappa": rspec.data["kappa"],
            "frequency": rspec.data["freq_min"],
            "pi_length": amp_rabi.data["pi_length"],
        }

        # Add error values
        qubit_dict.update(err_dict)

        # Add R² values for fit quality assessment
        r2_dict = {
            "t1_r2": t1.data["r2"],
            "t2r_r2": t2r.data["r2"],
            "t2e_r2": t2e.data["r2"],
            "rspec_r2": rspec.data["r2"],
            "amp_rabi_r2": amp_rabi.data["r2"],
        }
        qubit_dict.update(r2_dict)
    else:
        # Compile all measured parameters into a dictionary
        qubit_dict = {
            "t1": t1.data["new_t1_i"],
            "t2r": t2r.data["best_fit"][3],
            "f_ge": t2r.data["new_freq"],
        }
        if check_fid:
            qubit_dict["fidelity"] = shot.data["fids"][0]
            qubit_dict["phase"] = shot.data["angle"]

        # Add error values
        qubit_dict.update(err_dict)

        # Add R² values for fit quality assessment
        r2_dict = {
            "t1_r2": t1.data["r2"],
            "t2r_r2": t2r.data["r2"],
        }
        qubit_dict.update(r2_dict)

    # Round all values to 7 significant figures
    for key in qubit_dict:
        if isinstance(qubit_dict[key], (int, float)) and not np.isnan(qubit_dict[key]):
            qubit_dict[key] = round(qubit_dict[key], 7)

    return qubit_dict


def measure_cohere(qi, cfg_dict, update=True, display=False, max_t1=MAX_T1):
    """
    Measure and return key parameters for a qubit.

    This function performs a series of measurements to characterize a qubit's properties,
    including coherence times, frequencies, and readout fidelity.

    Parameters
    ----------
    qi : int
        Qubit index
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    update : bool, optional
        Whether to update the configuration file with new parameters
    display : bool, optional
        Whether to display plots during measurements
    max_t1 : float, optional
        Maximum T1 time in microseconds

    Returns
    -------
    dict
        Dictionary containing measured qubit parameters
    """
    cfg_path = cfg_dict["cfg_file"]

    # Step 1: T2 Ramsey
    t2r = meas.T2Experiment(
        cfg_dict, qi=qi, display=display, progress=False, style="fast"
    )
    if t2r.status and update:
        # Update qubit frequency and T2r time
        config.update_qubit(cfg_path, "f_ge", t2r.data["new_freq"], qi, verbose=False)
        config.update_qubit(
            cfg_path,
            "T2r",
            t2r.data["best_fit"][3],
            qi,
            rng_vals=[1, max_t1],
            sig=2,
            verbose=False,
        )

    if not t2r.status:
        recenter(qi, cfg_dict, t2r, update=update, display=display, max_t1=max_t1)

    # Step 2: T1 measurement
    t1 = meas.T1Experiment(
        cfg_dict, qi=qi, display=display, progress=False, style="fast"
    )
    # t1 = meas.T1Experiment(cfg_dict, qi=qi, display=display, progress=False, params={'span':300})
    if update:
        t1.update(rng_vals=[1, max_t1], verbose=False)

    if not t1.status:
        t1.data["new_t1_i"] = np.nan
        t1.display(debug=True)
        print("T1 failed")

    qubit_dict = set_up_dict(t1, t2r)

    return qubit_dict


def set_up_dict(t1, t2):
    err_dict = {
        "t2_err": np.sqrt(t2.data["fit_err_avgi"][3][3]),
        "fge_err": np.sqrt(t2.data["fit_err_avgi"][1][1]),
        "t1_err": np.sqrt(t1.data["fit_err_avgi"][2][2]),
    }
    # Compile all measured parameters into a dictionary
    qubit_dict = {
        "t1": t1.data["new_t1_i"],
        "t1_off": t1.data["best_fit"][0],
        "t1_amp": t1.data["best_fit"][1],
        "t2_off": t2.data["best_fit"][4],
        "t2_amp": t2.data["best_fit"][0],
        "t2": t2.data["best_fit"][3],
        "f_ge": t2.data["new_freq"],
    }
    # Add R² values for fit quality assessment
    r2_dict = {
        "t1_r2": t1.data["r2"],
        "t2_r2": t2.data["r2"],
    }

    # Add error values
    qubit_dict.update(err_dict)
    qubit_dict.update(r2_dict)

    # Round all values to 7 significant figures
    for key in qubit_dict:
        if isinstance(qubit_dict[key], (int, float)) and not np.isnan(qubit_dict[key]):
            qubit_dict[key] = round(qubit_dict[key], 7)

    return qubit_dict


def measure_setup(qi, cfg_dict):
    cfg_dict["cfg_file"] = None
    t1, t2r = measure_fast(qi, cfg_dict, i, t1, t2r)


def measure_fast(qi, cfg_dict, i, tdir, t1_val, t2_val, display=False,t2_type='T2r'):
    fname = str(Path(tdir) / f"t1_qubit{qi}_{i:05d}")
    t1 = meas.T1Experiment(
        cfg_dict,
        qi=qi,
        fname=fname,
        display=display,
        progress=False,
        style="fast",
        params={"span": 3.7 * t1_val},
    )

    fname = str(Path(tdir) / f"{t2_type}_qubit{qi}_{i:05d}")
    if t2_type=='T2r':
        params = {"span": 3.2 * t2_val, 'experiment_type': 'ramsey'}
    else:
        params = {"span": 3.2 * t2_val, 'experiment_type': 'echo'}
    t2 = meas.T2Experiment(
        cfg_dict,
        qi=qi,
        fname=fname,
        display=display,
        progress=False,
        style="fast",
        params=params,
    )

    qubit_dict = set_up_dict(t1, t2)
    return qubit_dict


def measure_fast2(qi, cfg_dict, i, t2e=None, t1=None, t1_val=30, t2_val=30):
    if t1 is None:
        t1 = meas.T1Experiment(
            cfg_dict, qi=qi, display=False, progress=False, style="fast"
        )
    else:
        t1.fname = str(Path(t1.fname).parent / f"t1_qubit{qi}_{i:%5d}")
        t1.span = 3.7 * t1_val

    if t2e is None:
        t2e = meas.T2Experiment(
            cfg_dict, qi=qi, display=False, progress=False, style="fast", params={'experiment_type':'echo'}
        )
        t2e.fname = str(Path(t2e.fname).parent / f"t2_qubit{qi}_{i:%5d}")
    t2e.span = 3 * t2_val

    return t1, t2e


def time_tracking(qubit_list, cfg_dict, total_time=12, display=False, fast=True):
    """
    Track qubit parameters over time.

    This function repeatedly measures qubit parameters over a specified time period
    and saves the results for tracking parameter drift.

    Parameters
    ----------
    qubit_list : list
        List of qubit indices to track
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    total_time : float, optional
        Total tracking time in hours
    display : bool, optional
        Whether to display plots during measurements

    Returns
    -------
    tuple
        (tracking_data, tracking_path) where tracking_data is a list of dictionaries
        containing the measured parameters for each qubit, and tracking_path is the
        path where the data is saved
    """
    # Create directory for tracking data
    base_path = Path(cfg_dict["expt_path"]).parents[1] / "Tracking"
    tracking_id = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}_{total_time:.1f}hrs'
    tracking_path = base_path / tracking_id
    tracking_path.mkdir(parents=True, exist_ok=True)
    (tracking_path / "images").mkdir()
    (tracking_path / "data").mkdir()
    cfg_dict = deepcopy(cfg_dict)  # Avoid modifying original cfg_dict
    cfg_dict["expt_path"] = str(tracking_path)

    # Initialize timing variables
    start_time = time.time()
    elapsed = 0
    i = 0

    # Run measurements until total_time is reached
    while elapsed < total_time:
        for j, qi in enumerate(qubit_list):
            # Measure current time
            tm = time.time()
            elapsed = (tm - start_time) / 3600
            print(f"Starting run {i}, for qubit {qi}. Time elapsed {elapsed:.2f} hrs")

            # Measure qubit parameters
            if fast:
                if i == 0:
                    t2 = 'T2e'
                    auto_cfg = config.load(cfg_dict["cfg_file"])
                    t1_val = auto_cfg["device"]["qubit"]["T1"][qi]
                    t2_val = auto_cfg["device"]["qubit"][t2][qi]
                else:
                    t1_val = tracking_data[j]["t1"][-1]
                    t2_val = tracking_data[j]["t2"][-1]
                    
                

                # cfg_dict['cfg_file']=None
                d = measure_fast(qi, cfg_dict, i, tracking_path, t1_val, t2_val, display=display, t2_type=t2)
                t1_val = d["t1"]
                t2_val = d["t2"]
                # d = measure_cohere(qi, cfg_dict, display=display)
                # d = measure_params(qi, cfg_dict, display=display, fast=True,  check_fid=False)
            else:
                d = measure_params(
                    qi, cfg_dict, display=display, fast=False, check_fid=False
                )
            d["time"] = tm
            d["elapsed"] = elapsed

            # Store data for this iteration
            if i == 0 and j == 0:
                # Initialize storage dictionary for each qubit on first iteration
                tracking_data = [
                    {key: [] for key in d.keys()} for _ in range(len(qubit_list))
                ]

            # Append values for each parameter
            for key, val in d.items():
                tracking_data[j][key].append(val)

        i += 1

        # Save tracking data to CSV files
        for j, qi in enumerate(qubit_list):
            csv_dir = base_path / "csv"
            csv_dir.mkdir(exist_ok=True)
            csv_path = str(csv_dir / f"{tracking_id}_qubit_{qi}_tracking.csv")

            # Convert tracking data dict to numpy arrays for saving
            data_arrays = {}
            for key in tracking_data[j].keys():
                data_arrays[key] = np.array(tracking_data[j][key])

            # Create header and data rows
            header = ",".join(data_arrays.keys())
            rows = np.vstack(list(data_arrays.values())).T
            # Future change to only add a new row 
            # Save to CSV
            np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")

    
    tt_stats = calc_stats(tracking_data)

    return tracking_data, tracking_path, tt_stats


def calc_stats(tracking_data): 

    for qubit in tracking_data:
        t1_arr = np.array(qubit['t1'])
        t2_arr = np.array(qubit['t2'])
        f_ge_arr = np.array(qubit['f_ge'])
        q1_val = np.pi * 2 * t1_arr * f_ge_arr/1e6
        q2_val = np.pi * 2 * t2_arr * f_ge_arr/1e6
        tphi = 1 / (1 / t2_arr - 1 / (2 * t1_arr))
        qubit['tphi'] = tphi
        qubit['q1'] = q1_val
        qubit['q2'] = q2_val

    tt_stats = {}

    for idx, qubit_data in enumerate(tracking_data):
        stats = {}
        for key, values in qubit_data.items():
            arr = np.array(values)
            stats[key] = {
                'mean': np.mean(arr),
                'max': np.max(arr),
                'std': np.std(arr)
            }
        tt_stats[idx] = stats

    return tt_stats


def recenter(qi, cfg_dict, t2r, update=True, display=False, max_t1=MAX_T1):
    # Try to recover from failed T2 measurement
    t2r.display(debug=True, refit=True)
    print(t2r.data["r2"])
    print(t2r.data["fit_err_par"])
    if not t2r.status:
        # Try to find qubit frequency with spectroscopy
        qubit_tuning.find_spec(qi, cfg_dict, start="fine")
        t2r = meas.T2Experiment(cfg_dict, qi=qi, display=display, progress=False)
        if t2r.status and update:
            config.update_qubit(
                cfg_dict["cfg_file"], "f_ge", t2r.data["new_freq"], qi, verbose=False
            )
            config.update_qubit(
                cfg_dict["cfg_file"],
                "T2r",
                t2r.data["best_fit"][3],
                qi,
                rng_vals=[1, max_t1],
                sig=2,
                verbose=False,
            )
            print("Recentered qubit frequency")

        if not t2r.status:
            # Handle persistently failed T2 measurement
            t2r.display(debug=True)
            t2r.data["best_fit"] = [np.nan, np.nan, np.nan, np.nan]
            t2r.data["new_freq"] = np.nan
            print("T2 Ramsey failed")
