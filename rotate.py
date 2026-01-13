import argparse
import torch

from pathlib import Path
from PIL import Image

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from zero123 import Zero123Pipeline


def rotate(
    start_img,
    out_folder="./result",
    a_step=3,
    e_step=0,
    i_steps=40,
    g_steps=None,
    device="cuda",
    size=320,
    cfg=7.0,
):
    pipeline = Zero123Pipeline.from_pretrained(
        "ashawkey/zero123-xl-diffusers",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipeline.to(device)
    start_img = Path(start_img)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    cond = Image.open(start_img).convert("RGB")
    w, h = cond.size
    if h != w:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        right = left + min_dim
        bot = top + min_dim
        cond = cond.crop((left, top, right, bot))
    if min(w, h) != size:
        cond = cond.resize((size, size))
    min_length = 4 if g_steps is None else len(str(g_steps))
    cond.save(out_folder / f"{0:0{min_length}d}.png")
    iters = 1
    print("Start generation!")
    while True:
        path = out_folder / f"{iters:0{min_length}d}.png"
        result = pipeline(
            cond,
            num_inference_steps=i_steps,
            azimuth=a_step,
            elevation=e_step,
            distance=0.0,
            guidance_scale=cfg,
            height=size,
            width=size,
        ).images[0]
        result.save(path)
        print(f"Saved image to {path}.")
        cond = Image.open(path).convert("RGB")
        iters += 1
        if g_steps is not None and iters >= g_steps:
            print("Complete!")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Zero12Collapse", description="Repeatedly rotates an image using Zero-123"
    )
    parser.add_argument("--img", help="Initial image")
    parser.add_argument("--out", help="Output folder", default="./result")
    parser.add_argument(
        "--azimuth", type=float, help="Azimuth rotation per step", default=3.0
    )
    parser.add_argument(
        "--elevation", type=float, help="Elevation rotation per step", default=0.0
    )
    parser.add_argument(
        "--inference", type=int, help="Diffusion inference steps", default=50
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of images to generate. If not specified, will generate endlessly.",
        default=None,
    )
    parser.add_argument("--device", help="Compute device", default="cuda")
    parser.add_argument(
        "--size",
        type=int,
        help="Size of image (must be square). Input images will be resized and center cropped.",
        default=512,
    )
    parser.add_argument(
        "--cfg",
        type=float,
        help="Guidance scale",
        default=3.0,
    )
    args = parser.parse_args()
    if not args.img:
        parser.print_help()
        exit()
    try:
        rotate(
            args.img,
            out_folder=args.out,
            a_step=args.azimuth,
            e_step=args.elevation,
            i_steps=args.inference,
            g_steps=args.steps,
            device=args.device,
            size=args.size,
            cfg=args.cfg,
        )
    except Exception as e:
        print("An error occured:")
        raise e
