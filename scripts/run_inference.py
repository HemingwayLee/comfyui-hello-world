#!/usr/bin/env python3
"""
Batch img2img inference for Qwen Image Edit (NSFW-v23) via ComfyUI API.

The Qwen Edit model takes input image(s) as conditioning (not as a noisy latent),
so it generates a new image guided by your input + text prompt.

Usage:
  # With server already running:
  python scripts/run_inference.py \\
      --input-dir input_images \\
      --output-dir output_images \\
      --prompt "make the sky dramatic at sunset" \\
      --nsfw

  # Auto-start server:
  python scripts/run_inference.py --auto-start ...
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
from PIL import Image, ImageOps

# ── ComfyUI server config ─────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 8188
BASE_URL = f"http://{HOST}:{PORT}"

# ── Model defaults ────────────────────────────────────────────────────────────
# v23 README: euler_ancestral/beta recommended, 4 steps, CFG=1
MODEL_NSFW = "Qwen-Rapid-AIO-NSFW-v23.safetensors"
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_SCHEDULER = "beta"
DEFAULT_STEPS = 4
DEFAULT_CFG = 1.0
DEFAULT_MAX_SIZE = 1024  # long-edge cap when using input size
ALIGN = 8               # VAE requires dimensions to be multiples of this

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def align(n: int) -> int:
    """Round n up to the nearest multiple of ALIGN."""
    return ((n + ALIGN - 1) // ALIGN) * ALIGN


def fit_dimensions(w: int, h: int, max_size: int) -> tuple[int, int]:
    """Scale (w, h) so the long edge <= max_size, then align both to ALIGN."""
    if max_size and max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = int(w * scale), int(h * scale)
    return align(w), align(h)


# ── Workflow builder ──────────────────────────────────────────────────────────

def build_workflow(
    prompt: str,
    input_filenames: list[str],
    output_prefix: str,
    model: str,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    width: int,
    height: int,
) -> dict:
    """
    Build a ComfyUI API workflow for Qwen Image Edit.

    input_filenames: 1–3 server-side filenames. The model sees them as
    Picture 1 / Picture 2 / Picture 3 in the prompt context.
    """
    if not 1 <= len(input_filenames) <= 3:
        raise ValueError(f"Expected 1–3 images, got {len(input_filenames)}")

    # Node IDs 10, 11, 12 are reserved for LoadImage nodes (pictures 1–3)
    load_node_ids = ["10", "11", "12"]

    positive_inputs: dict = {
        "prompt": prompt,
        "clip": ["1", 1],
        "vae": ["1", 2],
    }
    for i, filename in enumerate(input_filenames):
        positive_inputs[f"image{i + 1}"] = [load_node_ids[i], 0]

    workflow: dict = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        "3": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": positive_inputs,
        },
        "4": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": "", "clip": ["1", 1], "vae": ["1", 2]},
        },
        "9": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["9", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "5": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["2", 0], "vae": ["1", 2]},
        },
        "6": {
            "class_type": "SaveImage",
            "inputs": {"images": ["5", 0], "filename_prefix": output_prefix},
        },
    }

    for i, filename in enumerate(input_filenames):
        workflow[load_node_ids[i]] = {
            "class_type": "LoadImage",
            "inputs": {"image": filename, "upload": "image"},
        }

    return workflow


# ── ComfyUI API helpers ───────────────────────────────────────────────────────

def server_ready() -> bool:
    try:
        requests.get(f"{BASE_URL}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def upload_image(path: Path) -> str:
    """Upload image to ComfyUI input folder; returns server-side filename."""
    with open(path, "rb") as f:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        resp = requests.post(
            f"{BASE_URL}/upload/image",
            files={"image": (path.name, f, mime)},
        )
    resp.raise_for_status()
    return resp.json()["name"]


def queue_prompt(workflow: dict) -> str:
    """Submit workflow to the ComfyUI queue; returns prompt_id."""
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    resp = requests.post(f"{BASE_URL}/prompt", json=payload)
    if not resp.ok:
        detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        raise RuntimeError(f"Queue failed ({resp.status_code}): {detail}")
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 600) -> dict:
    """Poll /history until the prompt finishes; raises on timeout or error."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{BASE_URL}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise RuntimeError(f"ComfyUI error: {messages}")
            return entry
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")


def download_outputs(result: dict, out_dir: Path) -> list[Path]:
    """Fetch all generated images from ComfyUI and save to out_dir."""
    saved = []
    for node_out in result.get("outputs", {}).values():
        for img_meta in node_out.get("images", []):
            params = {
                "filename": img_meta["filename"],
                "subfolder": img_meta.get("subfolder", ""),
                "type": img_meta.get("type", "output"),
            }
            resp = requests.get(f"{BASE_URL}/view", params=params)
            resp.raise_for_status()
            dest = out_dir / img_meta["filename"]
            dest.write_bytes(resp.content)
            saved.append(dest)
    return saved


def open_with_exif(path: Path) -> Image.Image:
    """Open an image and apply EXIF orientation so .size matches the displayed dimensions."""
    return ImageOps.exif_transpose(Image.open(path))


def upscale_to(paths: list[Path], target_w: int, target_h: int) -> None:
    """Resize images on disk to (target_w, target_h) in-place using Lanczos."""
    for p in paths:
        with open_with_exif(p) as im:
            if im.size == (target_w, target_h):
                continue
            resized = im.resize((target_w, target_h), Image.LANCZOS)
            resized.save(p)


# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_server(project_root: Path) -> subprocess.Popen:
    python = project_root / "venv" / "bin" / "python"
    main_py = project_root / "ComfyUI" / "main.py"
    cmd = [
        str(python), str(main_py),
        "--use-pytorch-cross-attention",
        "--listen", HOST,
        "--port", str(PORT),
    ]
    print(f"Starting ComfyUI: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(project_root / "ComfyUI"))
    print("Waiting for server to be ready", end="", flush=True)
    for _ in range(90):
        if server_ready():
            print(" ✓")
            return proc
        print(".", end="", flush=True)
        time.sleep(2)
    proc.terminate()
    raise RuntimeError("ComfyUI failed to start within 3 minutes")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch img2img with Qwen Image Edit v23 via ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", type=Path,
                     help="Batch mode: directory of images, each processed independently")
    src.add_argument("--images", nargs="+", metavar="IMG", type=Path,
                     help="Multi-image mode: 1–3 images passed together as Picture 1 / Picture 2 / Picture 3")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to save generated images")
    p.add_argument("--prompt", required=True,
                   help="Edit instruction — use 'Picture 1', 'Picture 2', 'Picture 3' to reference images")
    p.add_argument("--nsfw", action="store_true",
                   help="Enable NSFW model (Qwen-Rapid-AIO-NSFW-v23)")
    p.add_argument("--model", default=None,
                   help="Override checkpoint name (as seen by ComfyUI)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help=f"Sampling steps (default: {DEFAULT_STEPS})")
    p.add_argument("--cfg", type=float, default=DEFAULT_CFG,
                   help=f"CFG scale (default: {DEFAULT_CFG}, keep at 1 for this model)")
    p.add_argument("--sampler", default=DEFAULT_SAMPLER,
                   choices=["euler_ancestral", "euler", "sa_solver", "lcm",
                            "er_sde", "dpm_2_ancestral", "dpmpp_2m"],
                   help=f"Sampler (default: {DEFAULT_SAMPLER})")
    p.add_argument("--scheduler", default=DEFAULT_SCHEDULER,
                   choices=["beta", "normal", "sgm_uniform", "simple", "karras"],
                   help=f"Noise scheduler (default: {DEFAULT_SCHEDULER})")
    p.add_argument("--seed", type=int, default=-1,
                   help="Seed (-1 = random per image)")
    p.add_argument("--width", type=int, default=None,
                   help="Fix output width in pixels (overrides input-size mode)")
    p.add_argument("--height", type=int, default=None,
                   help="Fix output height in pixels (overrides input-size mode)")
    p.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE,
                   help=f"Cap the long edge when using input size (default: {DEFAULT_MAX_SIZE}, 0 = no cap)")
    p.add_argument("--upscale-to-input", action="store_true",
                   help="Resize output back to the original input image size after generation")
    p.add_argument("--auto-start", action="store_true",
                   help="Auto-start ComfyUI server if not already running")
    p.add_argument("--timeout", type=int, default=600,
                   help="Seconds to wait per image (default: 600)")
    return p.parse_args()


def resolve_dimensions(img_paths: list[Path], args) -> tuple[int, int]:
    """Pick output width/height from args or the first image's size."""
    if args.width and args.height:
        return align(args.width), align(args.height)
    with open_with_exif(img_paths[0]) as im:
        src_w, src_h = im.size
    w = args.width or src_w
    h = args.height or src_h
    return fit_dimensions(w, h, args.max_size)


def run_job(img_paths: list[Path], output_prefix: str, args, model: str) -> list[Path]:
    """Upload images, queue a single workflow, return saved output paths."""
    with open_with_exif(img_paths[0]) as im:
        orig_w, orig_h = im.size

    width, height = resolve_dimensions(img_paths, args)
    seed = args.seed if args.seed >= 0 else int(time.time() * 1000) & 0xFFFFFFFF

    server_names = [upload_image(p) for p in img_paths]

    workflow = build_workflow(
        prompt=args.prompt,
        input_filenames=server_names,
        output_prefix=output_prefix,
        model=model,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
        scheduler=args.scheduler,
        seed=seed,
        width=width,
        height=height,
    )
    prompt_id = queue_prompt(workflow)
    print(f"queued ({width}×{height} seed={seed})", end="  ", flush=True)

    result = wait_for_completion(prompt_id, timeout=args.timeout)
    saved = download_outputs(result, args.output_dir)

    if args.upscale_to_input and (orig_w, orig_h) != (width, height):
        upscale_to(saved, orig_w, orig_h)
        print(f"upscaled to {orig_w}×{orig_h}", end="  ", flush=True)

    return saved


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).parent.parent

    # Resolve model
    model = args.model or MODEL_NSFW
    if not args.nsfw and not args.model:
        print(f"Note: using NSFW model by default ({MODEL_NSFW}).")
        print("      Pass --nsfw to explicitly select it.")

    # Collect jobs
    if args.images:
        if len(args.images) > 3:
            sys.exit("Error: --images accepts at most 3 images")
        for p in args.images:
            if not p.is_file():
                sys.exit(f"Error: image not found: {p}")
        jobs = [args.images]  # single job with all images together
        mode = f"multi-image ({len(args.images)} pictures)"
    else:
        in_dir: Path = args.input_dir
        if not in_dir.is_dir():
            sys.exit(f"Error: --input-dir '{in_dir}' not found")
        all_images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not all_images:
            sys.exit(f"No images found in {in_dir}")
        jobs = [[p] for p in all_images]  # one job per image
        mode = f"batch ({len(jobs)} images)"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure server
    server_proc = None
    if not server_ready():
        if args.auto_start:
            server_proc = start_server(project_root)
        else:
            sys.exit(
                f"ComfyUI not running at {BASE_URL}\n"
                "  Start it:  source .venv/bin/activate && "
                "python ComfyUI/main.py --use-pytorch-cross-attention\n"
                "  Or pass:   --auto-start"
            )

    size_desc = f"{args.width}×{args.height}" if (args.width and args.height) else f"input size (max-edge {args.max_size or 'unlimited'})"
    print(f"\nModel   : {model}")
    print(f"Mode    : {mode}")
    print(f"Sampler : {args.sampler}/{args.scheduler}  steps={args.steps}  cfg={args.cfg}")
    print(f"Prompt  : {args.prompt}")
    print(f"Size    : {size_desc}")
    print(f"Output  : {args.output_dir}\n")

    try:
        for idx, img_paths in enumerate(jobs, 1):
            label = " + ".join(p.name for p in img_paths)
            print(f"[{idx}/{len(jobs)}] {label}", end="  ", flush=True)

            saved = run_job(img_paths, f"qwen_{img_paths[0].stem}", args, model)
            for p in saved:
                print(f"→ {p.name}", end="  ")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if server_proc is not None:
            print("Stopping ComfyUI server...")
            server_proc.terminate()
            server_proc.wait()

    print(f"\nDone. Results in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
