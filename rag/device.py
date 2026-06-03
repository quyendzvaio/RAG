import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceChoice:
    device: str
    reason: str
    gpu_name: str = ""
    compute_capability: str = ""
    total_memory_mb: int = 0


def choose_torch_device(env_var: str, default: str = "auto") -> DeviceChoice:
    requested = os.getenv(env_var, default).strip().lower()
    if requested in {"cpu", "mps"}:
        return DeviceChoice(device=requested, reason=f"{env_var}={requested}")

    if requested in {"cuda", "gpu"} and _force_cuda():
        return DeviceChoice(device="cuda", reason=f"{env_var}={requested}, FORCE_CUDA=1")

    try:
        import torch
    except Exception as exc:
        return DeviceChoice(device="cpu", reason=f"torch chua san sang: {exc}")

    if requested in {"cuda", "gpu"} and not torch.cuda.is_available():
        return DeviceChoice(device="cpu", reason="torch khong thay CUDA kha dung")

    if requested not in {"auto", "cuda", "gpu"}:
        return DeviceChoice(device="cpu", reason=f"{env_var}={requested} khong ho tro, fallback CPU")

    if not torch.cuda.is_available():
        return DeviceChoice(device="cpu", reason="CUDA khong kha dung")

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_memory_mb = int(props.total_memory / (1024 * 1024))
    gpu_name = props.name

    if major < 5:
        return DeviceChoice(
            device="cpu",
            reason=(
                "GPU qua cu cho PyTorch/sentence-transformers hien dai "
                f"(compute capability {major}.{minor}); dung CPU de tranh loi CUDA kernel"
            ),
            gpu_name=gpu_name,
            compute_capability=f"{major}.{minor}",
            total_memory_mb=total_memory_mb,
        )

    if total_memory_mb < 3000:
        return DeviceChoice(
            device="cpu",
            reason=f"VRAM chi {total_memory_mb}MB, uu tien CPU de tranh out-of-memory",
            gpu_name=gpu_name,
            compute_capability=f"{major}.{minor}",
            total_memory_mb=total_memory_mb,
        )

    return DeviceChoice(
        device="cuda",
        reason="CUDA kha dung",
        gpu_name=gpu_name,
        compute_capability=f"{major}.{minor}",
        total_memory_mb=total_memory_mb,
    )


def nvidia_smi_summary() -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _force_cuda() -> bool:
    return os.getenv("FORCE_CUDA", "false").strip().lower() in {"1", "true", "yes"}
