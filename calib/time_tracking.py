import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ..helpers import config, rfboard
import slab_qick_calib.experiments as meas
from . import qubit_tuning

# Configure matplotlib
plt.rcParams["legend.handlelength"] = 0.5

# Default parameters
MAX_T1 = 500  # Maximum T1 time in microseconds
DEFAULT_MIN_T1 = 1.0  # Default minimum T1 time in microseconds
DEFAULT_MIN_T2 = 1.0  # Default minimum T2 time in microseconds
MAX_ERR = 1  # Maximum acceptable error in fits
MIN_R2 = 0.35  # Minimum acceptable R² value for fits
TOL = 0.3  # Tolerance for parameter convergence


def get_min_t1(cfg, qi):
    """Get per-qubit minimum T1 from config, falling back to DEFAULT_MIN_T1."""
    try:
        return cfg.device.qubit.min_T1[qi]
    except (AttributeError, KeyError, IndexError):
        return DEFAULT_MIN_T1


def get_min_t2(cfg, qi):
    """Get per-qubit minimum T2 from config, falling back to DEFAULT_MIN_T2."""
    try:
        return cfg.device.qubit.min_T2[qi]
    except (AttributeError, KeyError, IndexError):
        return DEFAULT_MIN_T2


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
    auto_cfg = config.load(cfg_path)
    min_t1 = get_min_t1(auto_cfg, qi)
    min_t2 = get_min_t2(auto_cfg, qi)
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
            rng_vals=[min_t2, max_t1],
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
        t1.update(rng_vals=[min_t1, max_t1], verbose=False)

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
                rng_vals=[min_t2, max_t1],
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
    auto_cfg = config.load(cfg_path)
    min_t1 = get_min_t1(auto_cfg, qi)
    min_t2 = get_min_t2(auto_cfg, qi)

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
            rng_vals=[min_t2, max_t1],
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
        t1.update(rng_vals=[min_t1, max_t1], verbose=False)

    if not t1.status:
        t1.data["new_t1_i"] = np.nan
        t1.display(debug=True)
        print("T1 failed")

    if t1 is None or t2r is None:        
        qubit_dict = {}
    else:
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
    ramsey_freq = 1.5 / t2_val
    if t2_type=='T2r':
        params = {"span": 3.2 * t2_val, 'experiment_type': 'ramsey', 'ramsey_freq': ramsey_freq}
    else:
        params = {"span": 3.2 * t2_val, 'experiment_type': 'echo', 'ramsey_freq': ramsey_freq}
    
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


def time_tracking(qubit_list, cfg_dict, total_time=12, display=False, fast=True, bf_client=None, soc=None, t1_max=None, t2_max=None):
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
    soc : object, optional
        QICK SoC object. If provided, calls rfboard.activate_qubit_rf()
        before each qubit's measurements to switch RF filters/attenuators.
    t1_max : float, optional
        Maximum allowed T1 value (µs). Measured values exceeding this are clamped.
    t2_max : float, optional
        Maximum allowed T2 value (µs). Measured values exceeding this are clamped.

    Returns
    -------
    tuple
        (tracking_data, tracking_path) where tracking_data is a list of dictionaries
        containing the measured parameters for each qubit, and tracking_path is the
        path where the data is saved
    """
    # Create directory for tracking data
    base_path = Path(cfg_dict["expt_path"]).parent / "Tracking"
    tracking_id = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}_{total_time:.1f}hrs'
    tracking_path = base_path / tracking_id
    tracking_path.mkdir(parents=True, exist_ok=True)
    (base_path / "images").mkdir(exist_ok=True)
    (tracking_path / "data").mkdir()
    cfg_dict = deepcopy(cfg_dict)  # Avoid modifying original cfg_dict
    cfg_dict["expt_path"] = str(tracking_path)

    # Initialize timing variables
    start_time = time.time()
    elapsed = 0
    i = 0

    # Run measurements until total_time is reached
    while elapsed < total_time:
        # Read MXC temperature and pulsetube status at start of each iteration
        mxc_temp = np.nan
        pulsetube_on = np.nan
        if bf_client is not None:
            try:
                mxc_temp = bf_client.get_mxc_temperature()
            except Exception as e:
                print(f"Warning: failed to read MXC temperature: {e}")
            try:
                pulsetube_on = float(bf_client.is_pulsetube_on())
            except Exception as e:
                print(f"Warning: failed to read pulsetube status: {e}")

        # Track which qubits succeeded this iteration (for CSV update)
        iter_success = [True] * len(qubit_list)

        for j, qi in enumerate(qubit_list):
            # Measure current time
            tm = time.time()
            elapsed = (tm - start_time) / 3600
            print(f"Starting run {i}, for qubit {qi}. Time elapsed {elapsed:.2f} hrs")

            # Switch RF board to this qubit's settings
            if soc is not None:
                auto_cfg = config.load(cfg_dict["cfg_file"])
                auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_dict["cfg_file"])

            # Measure qubit parameters
            if fast:
                if i == 0:
                    t2 = 'T2r'
                    if soc is None:
                        auto_cfg = config.load(cfg_dict["cfg_file"])
                    t1_val = auto_cfg["device"]["qubit"]["T1"][qi]
                    t2_val = auto_cfg["device"]["qubit"][t2][qi]
                else:
                    t1_val = tracking_data[j]["t1"][-1]
                    t2_val = tracking_data[j]["t2"][-1]

                try:
                    d = measure_fast(qi, cfg_dict, i, tracking_path, t1_val, t2_val, display=display, t2_type=t2)
                    if not d:
                        raise RuntimeError("Measurement returned empty result")
                    t1_val = d["t1"]
                    t2_val = d["t2"]
                except Exception as e:
                    print(f"Measurement failed for qubit {qi} on run {i}: {e}")
                    print("Using previous t1/t2 values for next iteration")
                    iter_success[j] = False
                    continue
            else:
                try:
                    d = measure_params(
                        qi, cfg_dict, display=display, fast=False, check_fid=False
                    )
                except Exception as e:
                    print(f"Measurement failed for qubit {qi} on run {i}: {e}")
                    iter_success[j] = False
                    continue

            # Clamp t1/t2 to max values if specified
            if t1_max is not None and d["t1"] > t1_max:
                d["t1"] = t1_max
            if t2_max is not None and d["t2"] > t2_max:
                d["t2"] = t2_max

            d["time"] = tm
            d["elapsed"] = elapsed
            if bf_client is not None:
                d["mxc_temp"] = mxc_temp
                d["pulsetube_on"] = pulsetube_on

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

        # Save tracking data to CSV files (only for qubits that succeeded)
        for j, qi in enumerate(qubit_list):
            if not iter_success[j]:
                continue

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
            np.savetxt(csv_path, rows, delimiter=",", header=header, comments="", fmt="%.5f")

    tt_stats = calc_stats(tracking_data)

    return tracking_data, tracking_path, tt_stats, tracking_id


def load_tracking(csv_dir, tracking_id=None, qubit_list=None):
    """
    Load tracking data from CSV files saved by time_tracking.

    Parameters
    ----------
    csv_dir : str or Path
        Path to the csv directory containing tracking CSV files
    tracking_id : str, optional
        Identifier for a specific run (e.g. "2026_02_09_14_30_12.0hrs").
        If None, uses the most recent run.
    qubit_list : list, optional
        List of qubit indices to load. If None, inferred from filenames.

    Returns
    -------
    tuple
        (tracking_data, csv_dir, tt_stats) matching the time_tracking return format
    """
    import re

    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("*_qubit_*_tracking.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No tracking CSV files found in {csv_dir}")

    # Group files by tracking_id (everything before '_qubit_')
    runs = {}
    for f in csv_files:
        match = re.match(r"(.+)_qubit_(\d+)_tracking\.csv", f.name)
        if match:
            tid, qi = match.group(1), int(match.group(2))
            runs.setdefault(tid, []).append((qi, f))

    if tracking_id is None:
        tracking_id = sorted(runs.keys())[-1]
    if tracking_id not in runs:
        available = ", ".join(sorted(runs.keys()))
        raise ValueError(
            f"tracking_id '{tracking_id}' not found. Available: {available}"
        )

    files = sorted(runs[tracking_id], key=lambda x: x[0])

    if qubit_list is not None:
        files = [(qi, f) for qi, f in files if qi in qubit_list]
    else:
        qubit_list = [qi for qi, _ in files]

    tracking_data = []
    for qi, fpath in files:
        data = np.genfromtxt(fpath, delimiter=",", names=True)
        qubit_dict = {name: data[name] for name in data.dtype.names}
        tracking_data.append(qubit_dict)

    tt_stats = calc_stats(tracking_data)

    # Ensure images directory exists in Tracking folder
    tracking_base = csv_dir.parent  # csv_dir is Tracking/csv, so parent is Tracking
    (tracking_base / "images").mkdir(exist_ok=True)

    # Report when CSVs were last updated
    most_recent = max(fpath.stat().st_mtime for _, fpath in files)
    last_modified = datetime.fromtimestamp(most_recent)
    elapsed = datetime.now() - last_modified
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes = remainder // 60
    print(
        f"Loaded tracking_id='{tracking_id}', qubits={qubit_list}\n"
        f"  CSV last updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({hours}h {minutes}m ago)"
    )
    return tracking_data, str(csv_dir), tt_stats, tracking_id


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
            auto_cfg = config.load(cfg_dict["cfg_file"])
            min_t2 = get_min_t2(auto_cfg, qi)
            config.update_qubit(
                cfg_dict["cfg_file"], "f_ge", t2r.data["new_freq"], qi, verbose=False
            )
            config.update_qubit(
                cfg_dict["cfg_file"],
                "T2r",
                t2r.data["best_fit"][3],
                qi,
                rng_vals=[min_t2, max_t1],
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
