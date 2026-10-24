"""Print the board's readout/generator map. Read-only: no run folder, no rfboard, no JPA."""

from qick import QickConfig

from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config

from common import REPO_DIR, parse_args


def adc_port(soccfg, adcname):
    """RF-board port for an ADC tile/block, ZCU216 numbering."""
    tile, block = [int(c) for c in adcname]
    return 4 * (tile - 1) + block


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    args = parse_args(__doc__, config=CONFIG)

    auto_cfg = config.load(str(REPO_DIR / "configs" / args.config))
    im = InstrumentManager(ns_address=auto_cfg["aliases"]["ip"], port=args.port)
    soc = im[auto_cfg["aliases"]["soc"]]
    soccfg = QickConfig(soc.get_cfg())

    print(f"\n{'idx':>4} {'adc':>5} {'port':>5} {'ro_type':<24} {'pfb_nch':>8} "
          f"{'f_dds':>10} {'tproc_in':>9} {'config':>8} "
          f"{'avg_maxlen':>11} {'buf_maxlen':>11}")
    print("-" * 106)
    by_adc = {}
    for i, r in enumerate(soccfg["readouts"]):
        name = r["adc"]
        by_adc.setdefault(name, []).append(i)
        print(f"{i:>4} {name:>5} {adc_port(soccfg, name):>5} {r['ro_type']:<24} "
              f"{str(r.get('pfb_nch', '-')):>8} {r['f_dds']:>10.3f} "
              f"{r['tproc_ch']:>9} "
              f"{'tProc' if 'tproc_ctrl' in r else 'PYNQ':>8} "
              f"{r.get('avg_maxlen', '-'):>11} {r.get('buf_maxlen', '-'):>11}")

    # avg_maxlen bounds shots x reads-per-shot on a channel. The while-loop
    # feedback channel fires a variable number of times per shot, so it is
    # the tightest constraint on `shots` once reset_mode is "while".
    print("\nreadout indices grouped by physical ADC:")
    for name, idxs in sorted(by_adc.items()):
        n = len(idxs)
        note = " <- several indices on one input: usable as a feedback channel" if n > 1 else ""
        print(f"  ADC{adc_port(soccfg, name)} (adc {name}): indices {idxs}{note}")

    print("\nconfig says:")
    ro = auto_cfg.hw.soc.adcs.readout
    for qi, (ch, ty) in enumerate(zip(ro.ch, ro.type)):
        adcname = soccfg["readouts"][ch]["adc"]
        print(f"  qi={qi}: readout index {ch} -> ADC{adc_port(soccfg, adcname)}, "
              f"type '{ty}' (firmware: "
              f"{'tProc' if 'tproc_ctrl' in soccfg['readouts'][ch] else 'PYNQ'}-configured)")

    print("\ngenerators:")
    for i, g in enumerate(soccfg["gens"]):
        print(f"  {i:>3}: {g['type']:<22} fs={g['fs']:9.3f}  "
              f"mixer={'yes' if g.get('has_mixer') else 'no'}")

    if "extra_description" in soccfg._cfg:
        print("\n" + "\n".join(soccfg["extra_description"]))

    # dreg_qty bounds add_reg (the while-loop counters); the dmem depth bounds
    # how many per-shot values 42_reset_while_loop can store.
    tproc = soccfg._cfg.get("tprocs", [{}])[0]
    print("")
    print("tProc:")
    for k in sorted(tproc):
        print(f"  {k} = {tproc[k]}")
