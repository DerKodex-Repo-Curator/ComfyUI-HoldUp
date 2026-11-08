import time
import sys

try:
    import GPUtil
except ImportError:  # a venv without pip cannot install it
    GPUtil = None
from comfy.comfy_types import IO


class HoldUp:
    """
    A passthrough node that pauses workflow execution to wait for GPU cooling
    and/or a fixed time delay. Accepts any input type and returns it unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (IO.ANY, {}),
                "use_waitTemperature": ("BOOLEAN", {"default": True}),
                "waitTemperature": ("INT", {"default": 50, "min": 45, "max": 90}),
                "waitSeconds": ("INT", {"default": 0, "min": 0, "max": 120}),
            },
            "optional": {
                "policy_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "execute_cool_down"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute to ensure wait happens
        return float("nan")

    def _display_temperature_progress(self, current_temp, target_temp, initial_peak_temp, bar_length=50):
        """
        Displays a text-based progress bar for GPU temperature cooldown.

        Args:
            current_temp (float): The current highest temperature of the GPUs.
            target_temp (float): The target cool-down temperature.
            initial_peak_temp (float): The highest temperature recorded when this
                                      cooldown cycle started or subsequently peaked.
            bar_length (int): The character length of the progress bar.
        """
        target_temp = float(target_temp)

        if initial_peak_temp <= target_temp:
            # Started at or below target
            progress = 1.0 if current_temp <= target_temp else 0.0
        elif current_temp <= target_temp:
            # Fully cooled
            progress = 1.0
        else:
            # Actively cooling from initial_peak_temp towards target_temp
            total_range_to_cool = initial_peak_temp - target_temp
            amount_cooled = initial_peak_temp - current_temp

            if total_range_to_cool <= 0:
                progress = 1.0 if current_temp <= target_temp else 0.0
            else:
                progress = amount_cooled / total_range_to_cool

            progress = max(0, min(1, progress))

        filled_length = int(bar_length * progress)
        bar_chars = '█' * filled_length + '-' * (bar_length - filled_length)

        status_message = f"***** Cooling GPUs: Peak: {initial_peak_temp:.1f}°C"
        full_line = f"\r{status_message} |{bar_chars}| {current_temp:.1f}°C / {target_temp:.1f}°C      "

        sys.stdout.write(full_line)
        sys.stdout.flush()

    def _temperatures(self):
        """(index, celsius) for every GPU, without requiring a Python package.

        nvidia-smi ships with the driver, so a project venv that cannot install
        gputil still gets a working gate. GPUtil is only a fallback.
        """
        import subprocess

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            rows = []
            for line in out.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    rows.append((int(parts[0]), float(parts[1])))
            if rows:
                return rows
        except Exception:
            pass
        try:
            import GPUtil

            return [(int(g.id), float(g.temperature)) for g in GPUtil.getGPUs()]
        except Exception:
            return []

    def _policy(self, policy_path):
        """Read the shared GPU policy, falling back to conservative defaults.

        The threshold is environmental, so it is not baked into a workflow: one
        file, shared by every project, changed without restarting anything.
        """
        import json
        import os

        path = policy_path or os.environ.get("GPU_POLICY_PATH") or "/app/server/data/gpu-policy.json"
        defaults = {"temp_limit_c": 50, "max_wait_s": 180, "sample_interval_s": 2, "fail_closed": True}
        try:
            with open(path) as fh:
                loaded = json.load(fh)
            merged = dict(defaults)
            merged.update({k: v for k, v in loaded.items() if k in defaults})
            return merged, path
        except (OSError, ValueError) as exc:
            print("[holdup] policy unavailable (%s); using defaults" % exc, file=sys.stderr)
            return defaults, path

    def _my_gpu_index(self):
        """The physical GPU this process will use.

        The launcher pins a project with --cuda-device N, so reading the process's
        own command line is the only reliable way to know which card is ours.
        """
        import os

        try:
            with open("/proc/self/cmdline", "rb") as fh:
                argv = fh.read().split(b"\x00")
            argv = [a.decode("utf-8", "replace") for a in argv if a]
            for i, arg in enumerate(argv):
                if arg == "--cuda-device" and i + 1 < len(argv):
                    return int(argv[i + 1])
                if arg.startswith("--cuda-device="):
                    return int(arg.split("=", 1)[1])
        except (OSError, ValueError):
            pass
        env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if env:
            try:
                return int(env.split(",")[0])
            except ValueError:
                pass
        try:
            import torch

            return int(torch.cuda.current_device())
        except Exception:
            return None

    def execute_cool_down(self, input, use_waitTemperature, waitTemperature, waitSeconds, policy_path=""):
        if use_waitTemperature:
            policy, path = self._policy(policy_path)
            # The policy wins over the widget: the threshold is environmental and
            # has to change in one place, not inside every saved workflow.
            # The policy is the ceiling; a graph may ask for a stricter one but
            # never a looser one. Ignoring the widget outright would make an
            # explicit 45 in a workflow silently mean 50.
            target = min(float(policy["temp_limit_c"]), float(waitTemperature))
            max_wait = float(policy["max_wait_s"])
            interval = float(policy["sample_interval_s"])
            fail_closed = bool(policy["fail_closed"])
            index = self._my_gpu_index()

            started = time.time()
            reported = False
            while True:
                temps = self._temperatures()
                if not temps:
                    print("[holdup] cannot read GPU temperature; skipping gate", file=sys.stderr)
                    break

                mine = [(i, c) for i, c in temps if index is None or i == int(index)]
                if not mine:
                    print("[holdup] GPU %s not found; failing closed" % index, file=sys.stderr)
                    if fail_closed:
                        raise RuntimeError("HoldUp: GPU %s is not visible to this process" % index)
                    break

                gpu_id, current = mine[0]
                if current <= target:
                    if reported:
                        print("\n[holdup] GPU %s at %.0fC <= %.0fC, continuing" % (gpu_id, current, target))
                    break

                elapsed = time.time() - started
                if elapsed >= max_wait:
                    message = "HoldUp: GPU %s stayed at %.0fC above %.0fC for %.0fs" % (
                        gpu_id, current, target, max_wait)
                    if fail_closed:
                        raise RuntimeError(message)
                    print("[holdup] %s; passing anyway (fail_closed=false)" % message, file=sys.stderr)
                    break

                if not reported:
                    print("[holdup] GPU %s at %.0fC > %.0fC; waiting up to %.0fs" % (
                        gpu_id, current, target, max_wait))
                    reported = True
                time.sleep(interval)

        if waitSeconds > 0:
            for i in range(int(waitSeconds), 0, -1):
                sys.stdout.write("\r***** Waiting: %d / %d seconds    " % (i, waitSeconds))
                sys.stdout.flush()
                time.sleep(1)
            print("\r***** Waited: %d seconds         " % waitSeconds)

        now = time.time()
        print("\r***** " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)))
        return (input,)
