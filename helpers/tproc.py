"""
tProc-level workarounds that have to be applied at the register level.

Wave-FIFO overflow
------------------
Each tProc output port feeds a wave FIFO (``soccfg['tprocs'][0]['fifo_depth']``,
512 entries on the current ZCU216 firmware). A ``WPORT_WR`` pushes one entry.

A program that plays a long pulse train from a *hardware* loop issues those
pushes far faster than the generator drains them: the CPMG loop in
``T2ProgramASM`` is 5 instructions, i.e. ~23 ns at f_core = 215.04 MHz, against
an inter-pulse period of 200-550 ns. The core therefore runs ~10x ahead and, at
roughly 500 pulses, fills the FIFO.

Whether a full FIFO stalls the core is a firmware config bit, ``TPROC_CFG[11]``
("WFIFO_Full Pause Core"). qick never sets it -- ``_init_firmware()`` leaves
``tproc_cfg = 0`` -- so by default the overflow is silent: writes issued after
the FIFO fills are dropped, including the readout pulse, and the acquisition
comes back as pure noise around zero in both I and Q (not a decayed signal, a
*missing* one).

Enabling the bit makes the core wait instead, which is the correct behaviour and
costs nothing when the FIFO is not full. Verified on qubit 1 of
S127_Quad4_chip7: CPMG at num_pi = 600 and 1000 goes from "readout amplitude
collapses to noise past a sharp threshold in wait time" to clean data over the
whole sweep, with no change to the pulse program.

The bit survives ``load_bin_program`` (``load_mem`` only masks the low 6 bits
with ``&= ~63``), but ``single_read``/``single_write`` assign ``tproc_cfg``
outright and would clear it, so it is re-checked before every acquisition.
"""

# tproc_cfg is register index 1; the exposed read/write take byte offsets.
TPROC_CFG_OFFSET = 1 * 4
WFIFO_PAUSE_BIT = 1 << 11


def wfifo_pause_enabled(soc):
    """Return True if the tProc is set to pause when a wave FIFO fills."""
    return bool(soc.tproc.read(TPROC_CFG_OFFSET) & WFIFO_PAUSE_BIT)


_warned = False


def enable_wfifo_pause(soc, verbose=False):
    """
    Set TPROC_CFG[11] so a full wave FIFO pauses the core instead of silently
    dropping pulses. Idempotent, and preserves the other bits.

    This is a workaround, so it never raises: if the register is not reachable
    (different firmware, non-tProc-v2 board, mocked soc) it warns once and
    leaves things alone.

    Returns True if the bit was newly set, False otherwise.
    """
    global _warned
    try:
        cfg = soc.tproc.read(TPROC_CFG_OFFSET)
        if cfg & WFIFO_PAUSE_BIT:
            return False
        soc.tproc.write(TPROC_CFG_OFFSET, cfg | WFIFO_PAUSE_BIT)
        if verbose:
            print(f"Set tproc_cfg WFIFO_Full Pause Core: 0x{cfg:X} -> "
                  f"0x{soc.tproc.read(TPROC_CFG_OFFSET):X}")
        return True
    except Exception as e:
        if not _warned:
            _warned = True
            print("Warning: could not set tproc_cfg WFIFO_Full Pause Core "
                  f"({type(e).__name__}: {e}). Pulse trains longer than about "
                  "500 pulses played from a hardware loop may silently lose "
                  "the readout pulse. See helpers/tproc.py.")
        return False


def max_safe_hw_loop_pulses(soccfg, loop_period_us, instr_per_iter=5):
    """
    Number of pulses a hardware loop can issue before the wave FIFO overflows,
    if WFIFO_Full Pause Core is not enabled.

    The core gains ``loop_period_us - instr_per_iter/f_core`` of lead per
    iteration, so FIFO occupancy after n pushes is
    ``n * (1 - t_exec/loop_period)``; overflow is when that reaches fifo_depth.

    Args:
        soccfg: QickConfig
        loop_period_us: scheduled duration of one loop iteration (us)
        instr_per_iter: tProc instructions in the loop body

    Returns:
        int pulse count, or None if the core cannot outrun the generator.
    """
    tp = soccfg["tprocs"][0]
    t_exec = instr_per_iter / tp["f_core"]
    if loop_period_us <= t_exec:
        return None
    return int(tp["fifo_depth"] / (1 - t_exec / loop_period_us))
