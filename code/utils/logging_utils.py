"""
Shared logging setup for infer.py and the utils it calls into.

setup_logging() does two things:
  1. Mirrors the process's stdout/stderr into a per-run log file inside the
     slide's output directory, so *everything* that would show up on the
     console - our own logging.info() calls as well as bare print()s from
     vendored utils or third-party libraries (trident/mmcv/mmdet/mmseg) - is
     also captured to disk, without duplicating lines between the console
     and the file.
  2. Points the root logger's handler at the (now-mirrored) stderr, so any
     module doing `logging.getLogger(__name__)` participates automatically.

This makes a direct `python infer.py ...` run produce the same kind of
persisted, complete log that batch_infer.py already gets "for free" via
subprocess stdout redirection.
"""
import logging
import os
import sys
import time


class _Tee:
    """Writes to multiple streams at once (e.g. the real console + a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        # Some libraries (ray/faulthandler, colorama's stream-wrapping proxy) introspect the
        # underlying file descriptor of stdout/stderr. Delegate to the first (real console)
        # stream, since that's the only one of self._streams backed by an actual OS fd.
        return self._streams[0].fileno()


def setup_logging(output_dir, log_filename="pipeline.log", level=logging.INFO):
    """
    Mirror stdout/stderr into `<output_dir>/<log_filename>` and configure the
    root logger to log through it. Safe to call once per process.

    Returns:
        str: path to the log file.
    """
    log_path = os.path.join(output_dir, log_filename)
    log_file = open(log_path, "w", buffering=1)  # line-buffered so tail -f works

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # pyvips propagates libvips' own informational chatter ("VIPS: residual scale ...",
    # "VIPS: vips__open_image_write: simple open") through Python logging at INFO level; it's
    # extremely high-volume (one line per tile write) and not actionable, so keep it at WARNING+.
    logging.getLogger("pyvips").setLevel(logging.WARNING)

    return log_path


class TimingTracker:
    """Tracks and logs start/end/elapsed time for each pipeline step, plus a final summary table."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.timings = {}
        self._start_times = {}

    def start_step(self, step_name):
        self._start_times[step_name] = time.perf_counter()
        self.logger.info(f"[STEP START] {step_name}")

    def end_step(self, step_name):
        if step_name not in self._start_times:
            self.logger.warning(f"No start time recorded for step: {step_name}")
            return
        elapsed = time.perf_counter() - self._start_times[step_name]
        self.timings[step_name] = elapsed
        minutes, seconds = divmod(elapsed, 60)
        self.logger.info(f"[STEP END] {step_name} - Elapsed: {int(minutes)}m {seconds:.2f}s")

    def log_summary(self):
        if not self.timings:
            self.logger.warning("No timing data available")
            return
        total_time = sum(self.timings.values())
        self.logger.info("=" * 70)
        self.logger.info("PIPELINE TIMING SUMMARY")
        self.logger.info("=" * 70)
        for i, (step_name, elapsed) in enumerate(self.timings.items(), 1):
            minutes, seconds = divmod(elapsed, 60)
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            self.logger.info(
                f"{i:2d}. {step_name:<45s} {int(minutes):2d}m {seconds:05.2f}s ({percentage:5.1f}%)"
            )
        self.logger.info("-" * 70)
        total_minutes, total_seconds = divmod(total_time, 60)
        self.logger.info(f"TOTAL TIME: {int(total_minutes)}m {total_seconds:.2f}s")
        self.logger.info("=" * 70)
