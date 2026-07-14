"""Convert RoboTwin HDF5 demonstrations to a local LeRobot dataset.

The source data is read-only.  Output is written below ``HF_LEROBOT_HOME`` so
that openpi's standard ``LeRobotDataset(repo_id)`` loader can consume it.

Example:
    uv run examples/robotwin/convert_robotwin_data_to_lerobot.py \
        --raw-dir /home/ubuntu/Workspace/RoboTwin/data/place_block_aba_grid/demo_clean \
        --repo-id local/robotwin_place_block_aba_grid
"""

import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate",
    "left_gripper", "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle",
    "right_wrist_rotate", "right_gripper",
]


def _decode_jpeg(value: np.bytes_) -> np.ndarray:
    """Decode the fixed-width, NUL-padded JPEG bytes stored by RoboTwin."""
    image = cv2.imdecode(np.frombuffer(bytes(value).rstrip(b"\0"), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode a RoboTwin JPEG frame")
    # LeRobot's image feature uses CHW; OpenCV decoded BGR is what RoboTwin's
    # existing pi05 conversion and runtime use, so retain that convention.
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def _episode_instruction(raw_dir: Path, episode_id: int) -> str:
    import json

    data = json.loads((raw_dir / "instructions" / f"episode{episode_id}.json").read_text())
    # Keep the first seen phrasing deterministic.  Other seen paraphrases can
    # be added in a future augmentation pass without changing raw trajectories.
    return data["seen"][0]


def convert(raw_dir: Path, repo_id: str, *, fps: int = 50, overwrite: bool = False) -> None:
    raw_dir = raw_dir.resolve()
    files = sorted((raw_dir / "data").glob("episode*.hdf5"), key=lambda p: int(p.stem.removeprefix("episode")))
    if not files:
        raise FileNotFoundError(f"No episode*.hdf5 files under {raw_dir / 'data'}")

    output_dir = HF_LEROBOT_HOME / repo_id
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        **{
            f"observation.images.{name}": {
                "dtype": "image", "shape": (3, 240, 320), "names": ["channels", "height", "width"]
            }
            for name in CAMERA_MAP
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="aloha",
        features=features,
        use_videos=False,
        image_writer_threads=12,
    )

    for episode_id, path in enumerate(files):
        with h5py.File(path, "r") as episode:
            state = episode["joint_action/vector"][:].astype(np.float32)
            # Model input at t predicts the robot's absolute joint target at
            # t+1. This matches RoboTwin's established pi05 preprocessing.
            for frame_id in range(len(state) - 1):
                frame = {
                    "observation.state": state[frame_id],
                    "action": state[frame_id + 1],
                    "task": _episode_instruction(raw_dir, episode_id),
                }
                for output_name, source_name in CAMERA_MAP.items():
                    frame[f"observation.images.{output_name}"] = _decode_jpeg(
                        episode["observation"][source_name]["rgb"][frame_id]
                    )
                dataset.add_frame(frame)
        dataset.save_episode()
        print(f"Converted episode {episode_id + 1}/{len(files)}")

    print(f"LeRobot dataset: {dataset.root}")
    print(f"Episodes: {dataset.num_episodes}; frames: {dataset.num_frames}")


if __name__ == "__main__":
    tyro.cli(convert)
