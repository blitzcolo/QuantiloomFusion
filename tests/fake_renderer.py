"""A stand-in for QuantiloomCli that fabricates valid capture outputs.

write_outputs produces the frame.metadata.json record and product files that
datagen.export_record.verify_capture accepts with pixels=False, so the whole
run/publish/resume/repair path can be exercised without the renderer or a GPU.
"""

import math
import tomllib
import uuid
from pathlib import Path

from quantiloom_fusion.core.hashing import digest
from quantiloom_fusion.core.jsonio import atomic_json
from quantiloom_fusion.core.paths import native_path
from quantiloom_fusion.core.tomlio import toml_text
from quantiloom_fusion.core.vec import add, cross, scale, unit


class FakeProcess:
    """A running render stand-in: pid, wait() -> exit code, terminate()."""

    def __init__(self, renderer, key, code, interrupt=False):
        # pid 0 is never alive(), so a stale lock holding it looks orphaned.
        self.pid = 0
        self._renderer = renderer
        self._key = key
        self._code = code
        self._interrupt = interrupt

    def wait(self):
        if self._interrupt:
            raise KeyboardInterrupt
        return self._code

    def terminate(self, timeout_s=10.0):
        self._renderer.terminated.append(self._key)


class FakeRenderer:
    """Records calls and fabricates a complete export record per capture."""

    def __init__(self, version="Quantiloom 0.7.0", exit_codes=None, interrupt=None):
        self._version = version
        self.exit_codes = {k: list(v) for k, v in (exit_codes or {}).items()}
        self.interrupt = set(interrupt or ())
        self.calls = []
        self.terminated = []

    def version(self):
        return self._version

    def start(self, job_toml, workdir, log):
        parts = Path(job_toml).parts
        i = parts.index("attempts")
        key = (parts[i + 1], parts[i + 2])
        self.calls.append(key)
        if key in self.interrupt:
            return FakeProcess(self, key, 0, interrupt=True)
        codes = self.exit_codes.get(key)
        code = codes.pop(0) if codes else 0
        if code:
            log.write(f"fake renderer: exit {code}\n")
            return FakeProcess(self, key, code)
        self.write_outputs(job_toml, workdir)
        return FakeProcess(self, key, code)

    def write_outputs(self, job_toml, workdir):
        """Write the export record + product files a real capture produces."""
        workdir = Path(workdir)
        job = tomllib.loads(native_path(job_toml).read_text(encoding="utf-8"))
        w, h = job["renderer"]["resolution"]
        f = h / (2 * math.tan(math.radians(job["camera"]["fov_y"]) / 2))
        intrinsics = [f, 0, w / 2, 0, f, h / 2, 0, 0, 1]
        # Invert the planner's projection of camera_to_world: look_at is
        # position + forward column, up is -down column.
        position = job["camera"]["position"]
        forward = unit(add(job["camera"]["look_at"], scale(position, -1)))
        down = scale(job["camera"]["up"], -1)
        right = unit(cross(down, forward))
        camera_to_world = [
            right[0], down[0], forward[0], position[0],
            right[1], down[1], forward[1], position[1],
            right[2], down[2], forward[2], position[2],
            0, 0, 0, 1,
        ]
        world_to_camera = [
            right[0], right[1], right[2], -(right[0] * position[0] + right[1] * position[1] + right[2] * position[2]),
            down[0], down[1], down[2], -(down[0] * position[0] + down[1] * position[1] + down[2] * position[2]),
            forward[0], forward[1], forward[2], -(forward[0] * position[0] + forward[1] * position[1] + forward[2] * position[2]),
            0, 0, 0, 1,
        ]
        geometry = {
            "camera_model": "brown_conrady",
            "camera_axes": "right_down_forward",
            "intrinsics": intrinsics,
            "camera_to_world": camera_to_world,
            "world_to_camera": world_to_camera,
            "distortion": {
                "model": "brown_conrady",
                "coefficient_order": "k1_k2_p1_p2_k3",
                # The lens the job asked for, so a co-view reference carries
                # the VIS distortion it was planned with.
                "coefficients": list(job["sensor"]["optics"]["projection"]["coefficients"]),
                "max_theta_radians": 1.5,
            },
            "width": w,
            "height": h,
            "pixel_center_offset": [0.5, 0.5],
            "pixel_origin": "top_left",
            "world_units_to_meters": 1.0,
        }
        # The real renderer echoes the resolved config back into provenance:
        # cfa and readout defaults come from the sensor preset, isp gains the
        # white balance and colour matrix it actually applied.
        sensor = {**job["sensor"]}
        sensor["cfa"] = {"generic_cmos": "rggb"}.get(sensor.get("preset"), "mono")
        sensor["readout"] = {"black_level_dn": 0, "electrons_per_dn": 1.0, **sensor["readout"]}
        isp = {**job.get("isp", {}), "white_balance": [1.0, 1.0, 1.0],
               "device_to_linear_srgb": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}
        sensor_toml = toml_text({"sensor": sensor, "isp": isp})
        record_id = uuid.uuid4().hex * 2
        replay = workdir / "frame.replay.toml"
        replay.write_text('frame = "replay"\n', encoding="utf-8", newline="\n")
        entries = [
            {
                "path": replay.name,
                "sha256": digest(replay),
                "size_bytes": replay.stat().st_size,
            }
        ]
        products = []
        semantics = {
            "frame_measurement.exr": ("radiance", "W/m^2/sr"),
            "frame_rawdn.exr": ("rawdn", "dn"),
            "frame_corrected.exr": ("radiance", "W/m^2/sr"),
            "frame_display.exr": ("display", "1"),
        }
        for name in (
            "frame_measurement.exr",
            "frame_rawdn.exr",
            "frame_corrected.exr",
            "frame_display.exr",
            "frame.exr",
            "frame_truth_camera_z.exr",
        ):
            path = workdir / name
            path.write_bytes(b"FAKE-EXR:" + name.encode() + b"\n")
            entry = {
                "path": name,
                "sha256": digest(path),
                "size_bytes": path.stat().st_size,
                "product_id": f"{record_id[:16]}-{name}",
                "description": {},
            }
            if name in semantics:
                kind, unit_name = semantics[name]
                entry["description"] = {
                    "width": w,
                    "height": h,
                    "channel_names": ["Y"],
                    "signal": {
                        "kind": kind,
                        "unit": unit_name,
                        "transfer": "linear",
                        "camera_cfa": "0",
                        "channels": ["Y"],
                        "colour_space": "none",
                        "storage": "float32",
                    },
                    "image_metadata": {},
                    "provenance": {
                        "camera_config_toml": sensor_toml,
                        "geometry": geometry,
                    },
                }
            products.append(entry)
        atomic_json(
            workdir / "frame.metadata.json",
            {
                "schema": "quantiloom.dataset.export",
                "schema_version": 2,
                "state": "complete",
                "capture_status": "complete",
                "record_id": record_id,
                "replay": entries[0],
                "products": products,
            },
        )
