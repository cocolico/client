"""Native absolute EEF + Sharpa qpos + head interface; no Aloha conversions."""
from dataclasses import dataclass
import numpy as np

HAND_JOINTS = ("thumb_CMC_FE","thumb_CMC_AA","thumb_MCP_FE","thumb_MCP_AA","thumb_IP",
               "index_MCP_FE","index_MCP_AA","index_PIP","index_DIP",
               "middle_MCP_FE","middle_MCP_AA","middle_PIP","middle_DIP",
               "ring_MCP_FE","ring_MCP_AA","ring_PIP","ring_DIP",
               "pinky_CMC","pinky_MCP_FE","pinky_MCP_AA","pinky_PIP","pinky_DIP")
FEATURE_NAMES = tuple(n for side in ("left","right") for n in
                      ([f"{side}_ee_{s}" for s in ("x","y","z","qx","qy","qz","qw")]
                       + [f"{side}_hand_{s}" for s in HAND_JOINTS])) + (
    "idx11_head_joint1","idx12_head_joint2","idx13_head_joint3")
CAMERA_KEYS = ("head_color","left_wrist_camera","right_wrist_camera")
IMAGE_MAP = dict(zip(CAMERA_KEYS,("base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb")))


def _vector(value, *, actions=False):
    a=np.asarray(value,dtype=np.float32)
    if (a.ndim!=2 or a.shape[1]!=61 or not len(a)) if actions else (a.shape!=(61,)):
        raise ValueError("Expected finite [chunk,61] actions" if actions else "Expected finite 61D state")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite state/action")
    return a.copy()


def _image(value):
    a=np.asarray(value)
    if a.ndim!=3:
        raise ValueError("Expected a three-dimensional RGB image")
    if a.shape[-1]!=3 and a.shape[0]==3:
        a=a.transpose(1,2,0)
    if a.shape[-1]!=3 or min(a.shape[:2])<1:
        raise ValueError("Expected RGB channels")
    if np.issubdtype(a.dtype,np.floating):
        if not np.isfinite(a).all() or np.min(a)<0 or np.max(a)>1:
            raise ValueError("Float RGB must be finite within [0,1]")
        a=(a*255).astype(np.uint8)
    elif a.dtype!=np.uint8:
        raise ValueError("RGB must be uint8 or normalized float")
    return np.ascontiguousarray(a)


@dataclass(frozen=True)
class SharpaInputs:
    camera_mode: str = "three"

    def __call__(self,data):
        images=data.get("images",{})
        if self.camera_mode not in ("three", "head"):
            raise ValueError(f"Unknown Sharpa camera mode: {self.camera_mode}")
        if self.camera_mode == "three" and set(images)!=set(CAMERA_KEYS):
            raise ValueError(f"All three Sharpa cameras are required: {CAMERA_KEYS}")
        if self.camera_mode == "head" and "head_color" not in images:
            raise ValueError("Head camera is required")
        active = CAMERA_KEYS if self.camera_mode == "three" else ("head_color",)
        selected = {IMAGE_MAP[source]:_image(images[source]) for source in active}
        # Keep the pretrained model's fixed input structure, but expose no wrist
        # pixels/tokens to attention in the head-only ablation.
        for source in CAMERA_KEYS:
            if source not in active:
                selected[IMAGE_MAP[source]] = np.zeros_like(selected["base_0_rgb"])
        out={"state":_vector(data["state"]),
             "image":selected,
             "image_mask":{dest:np.bool_(source in active) for source,dest in IMAGE_MAP.items()}}
        if "actions" in data:out["actions"]=_vector(data["actions"],actions=True)
        if "prompt" in data:out["prompt"]=data["prompt"]
        return out


@dataclass(frozen=True)
class SharpaOutputs:
    def __call__(self,data):
        return {"actions":_vector(data["actions"],actions=True)}

