# SO-100 Robot Hackathon: Setup, Calibration, Dataset Recording & Training

## Theme 4: Physical AI — Teach a Robot Arm with Imitation Learning

This is **Theme 4** of the hackathon. You will work with a real robotic arm, teach it tasks through physical demonstration, and integrate it with [Fanar](https://fanar.qa) — a large language model with Arabic and English support — to build a system that responds to natural language or speech commands.

**This theme is limited to 3 teams.**

---

### What You Are Building

The core idea is **imitation learning**: you physically demonstrate a task with the robot arm, record it, and train a neural network to reproduce that behavior. This gives the robot a motor skill.

**Fanar integration is required.** Every project in this theme must use Fanar in some meaningful way. Fanar is your interface between human intent and robot action — it can interpret speech, parse instructions, classify what it sees, or reason about what the robot should do next. How you connect them is up to your team.

Some example directions:

- **Pill sorting**: the user says "put the red pills in the left container and the blue pills in the right" — Fanar interprets the command, the robot identifies the pills by color and places them accordingly.
- **Document handling**: the robot reads a document under a camera; Fanar classifies it by language (Arabic vs. English), status (expired vs. not expired), or category (e.g. female vs. male patient records) and the robot routes it accordingly.
- **Object manipulation by description**: the user names an object or property in natural language; Fanar parses the intent and instructs the robot to retrieve or sort it.

The pattern every project should follow: **speech or text → Fanar → robot action**. The specific bridge you build between those steps is your design challenge.

---

### Key Concepts

**ACT — Action Chunking with Transformers**

ACT is the policy we recommend for this theme. It is a transformer-based model that learns to predict short sequences of actions (chunks) from camera images and joint states. It was designed specifically for robotic manipulation and works well with 50–150 demonstrations. You do not need to write any model code — LeRobot provides it out of the box via `--policy.type=act`.

**VLA — Vision-Language-Action Models**

VLA models extend the idea further: instead of conditioning only on images and joint states, they take a natural language instruction as an additional input and produce robot actions directly. This is what enables commands like "pick up the red block" without retraining for every new task. VLAs are larger and require more compute, but they are the natural fit for Fanar integration and are worth exploring if your team wants to go deeper. Ask an organizer about available VLA checkpoints.

---

### Logistics — Please Read Before Signing Up

**Sign up now.** Spots are allocated on a **first come, first served basis**. If your team is interested in this theme, let us know as soon as possible.

**Lab access is required.** The robot arm cannot leave the lab. Working on this theme means your team must physically come in to use it. We only have **one robot**, so we will coordinate a shared schedule and rotate access between the 3 teams. Let us know your availability and we will work out the slots together.

**VPN access for GPU training.** To SSH into the GPU workstation for training, you will need VPN access to the university network. If you do not already have this set up, **contact us immediately** — arranging VPN credentials takes time and must be done before the weekend.

---

### Further Reading

- [LeRobot: A Community-Driven Robotics Framework (official paper)](https://arxiv.org/pdf/2510.12403) — deep dive into datasets, policies, and training
- [Modern AI Robotics from First Principles](https://interlatent.com/blog/interlatent-modern-ai-robotics-first-principles) — conceptual foundation for understanding learning-based robot control
- [Do As I Do — imitation learning project](https://do-as-i-do.com) · [GitHub](https://github.com/malik-group/do-as-i-do) — a worked example of robot imitation learning, useful for understanding the full pipeline end-to-end
- [VLA SOTA](https://arxiv.org/pdf/2511.14759)

---

### This Guide

The rest of this document covers the technical setup: installing LeRobot, calibrating the arms, recording a dataset, and launching training on the shared GPU workstation. Follow the steps in order.

---

## Overview: Who Does What, Where

You will use **two machines**:

| Machine | What you do there |
|---|---|
| **Your laptop** | Connect to the robot arms, run calibration, teleoperate, record datasets |
| **GPU workstation** (SSH) | Train the policy on your recorded dataset |

USB serial devices (the robot arms) are only visible to the machine they are **physically plugged into**. Port detection and all robot control must run on your laptop.

---

## Part 1 — Install LeRobot on Your Laptop

### 1.1 Create a Python environment

Using conda:

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

Or using `uv`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 1.2 Install LeRobot with Feetech motor support

Clone the repository and install:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech]"
```

If you also plan to record datasets and use a camera, add the dataset extra:

```bash
pip install -e ".[feetech,dataset]"
pip install pynput        # enables keyboard start/stop control during recording
```

> `pynput` lets you press a key to start and stop each episode. Without it, LeRobot falls back to a fixed-duration timer per episode.

### 1.3 Verify the installation

```bash
python -c "import lerobot; print('LeRobot OK')"
```

You should also have these CLI commands available:

```text
lerobot-find-port
lerobot-find-cameras
lerobot-calibrate
lerobot-teleoperate
lerobot-record
lerobot-replay
lerobot-train
```

---

## Part 2 — Find the Robot Arm Ports

The SO-100 leader and follower arms each appear as a USB serial port. You need to identify which port belongs to which arm.

### 2.1 Plug in one arm at a time

Run the port-detection tool:

```bash
lerobot-find-port
```

When prompted:

1. It will ask you to unplug one USB cable.
2. Unplug the arm you want to identify.
3. Press Enter.
4. The tool prints the port for that arm.
5. Plug the arm back in, then repeat for the other arm.

### 2.2 Record your ports

Write down the result — you will use these values in every subsequent command.

| Arm | Port (example macOS) | Port (example Linux) |
|---|---|---|
| Leader | `/dev/tty.usbmodem<ID>` | `/dev/ttyUSB0` or `/dev/ttyACM0` |
| Follower | `/dev/tty.usbmodem<ID>` | `/dev/ttyUSB1` or `/dev/ttyACM1` |

> On Linux you may need to add yourself to the `dialout` group if you get a permission error:
> `sudo usermod -aG dialout $USER` (log out and back in after).

Choose ID strings for your arms (these are just labels, pick anything consistent):

```text
Leader ID:   leader_so100
Follower ID: follower_so100
```

Keep these IDs consistent — LeRobot stores calibration files keyed on them.

---

## Part 3 — Calibrate the Arms

Calibration records each motor's range of motion. Run it once per arm.

### 3.1 Calibrate the leader arm

```bash
lerobot-calibrate \
  --teleop.type=so100_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=leader_so100
```

Replace `<LEADER_PORT>` with your detected port (e.g. `/dev/tty.usbmodem58C10000001`).

**What happens during calibration:**

1. LeRobot asks you to move the arm to a **neutral middle pose**. A good starting position:
   - Shoulder pan: centered (not turned left or right)
   - Shoulder lift: mid-height
   - Elbow flex: half-bent (not fully straight or fully folded)
   - Wrist flex: roughly neutral
   - Gripper: half open
2. Press Enter to confirm the neutral pose.
3. LeRobot then asks you to move each joint slowly through its **full safe range** before pressing Enter. Do this for each joint it names.
   - Move slowly and deliberately.
   - Do not force past mechanical resistance.
   - Make sure each joint has a clearly different min and max value (move it far enough).

**Common error and fix:**

```text
ValueError: Some motors have the same min and max values: ['shoulder_pan']
```

This means a joint was not moved enough. Rerun calibration and sweep that joint fully through its range before pressing Enter.

### 3.2 Calibrate the follower arm

```bash
lerobot-calibrate \
  --robot.type=so100_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=follower_so100
```

Apply the same neutral-pose and full-range-sweep steps as above.

> **IMPORTANT — Do not touch the setup after calibration is complete.**
> Once a team has finished calibration, the physical environment (camera position, robot placement, object positions, lighting) must remain exactly as it is. Imitation learning is highly sensitive to environmental consistency — if the camera shifts even slightly, or objects are moved between sessions, the trained model will fail to generalize and the entire dataset becomes unreliable. If you need to adjust anything, inform an organizer first. This applies to all teams sharing the robot.

---

## Part 4 — Test Teleoperation

Before recording data, verify that the leader arm can control the follower arm:

```bash
lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=follower_so100 \
  --teleop.type=so100_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=leader_so100
```

Move the leader arm and confirm the follower mirrors it. If this works, your calibration is good.

Expected confirmation:

```text
[OK] Connected to leader arm
[OK] Connected to follower arm
[OK] Calibration files loaded
[OK] Follower tracking leader movements
```

Press `Ctrl+C` to stop.

---

## Part 5 — Find the Camera

```bash
lerobot-find-cameras
```

This lists all detected cameras with their OpenCV index. The first camera is typically index `0`. Note the index you want to use.

No separate camera calibration step is needed — LeRobot uses the camera directly as an observation source.

---

## Part 6 — Log In to Hugging Face

LeRobot uploads your dataset to [Hugging Face Hub](https://huggingface.co). You need an account and a write-access token.

1. Create a free account at https://huggingface.co if you do not have one.
2. Go to **Settings → Access Tokens** and create a token with **Write** permission.
3. Log in on your laptop:

```bash
huggingface-cli login
# or, if that command is unavailable:
hf auth login
```

Paste your token when prompted. Verify:

```bash
huggingface-cli whoami
# or:
hf auth whoami
```

Choose a `repo_id` for your dataset in the format `<your-hf-username>/<dataset-name>`, for example:

```text
yourname/so100_pick_place
```

---

## Part 7 — Record a Dataset

Now record teleoperation episodes with the camera. Replace the port, ID, camera index, and `repo_id` values with your own.

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=follower_so100 \
  --robot.cameras="{front: {type: opencv, index_or_path: <CAMERA_INDEX>, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=leader_so100 \
  --dataset.repo_id=<HF_USERNAME>/<DATASET_NAME> \
  --dataset.num_episodes=50
```

**Tips for a good dataset:**

- Pick a single, repeatable task (e.g. pick up a block and place it in a bin).
- Keep the camera and the object in the same position for every episode.
- Aim for at least **50 episodes**. More is better; 100+ is ideal for reliable training.
- Each episode should be one clean, complete demonstration of the task.
- Discard episodes where you made a mistake — quality matters more than quantity.

**What LeRobot does:**

- Saves episodes locally to `~/.cache/huggingface/lerobot/<hf-username>/<dataset-name>_<timestamp>/`
- Pushes the dataset to Hugging Face Hub at the end of the recording session.

**If the push fails** and you need to upload manually:

```python
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
local_path = Path("~/.cache/huggingface/lerobot/<hf-username>/<timestamped-folder>").expanduser()

api.create_repo("<HF_USERNAME>/<DATASET_NAME>", repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path=str(local_path),
    repo_id="<HF_USERNAME>/<DATASET_NAME>",
    repo_type="dataset"
)
```

---

## Part 8 — Train on the GPU Workstation

SSH into the shared GPU workstation (you will be given the address and credentials):

```bash
ssh <your-username>@<workstation-address>
```

### 8.1 Install LeRobot on the workstation

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech,dataset]"
```

Log in to Hugging Face on the workstation as well:

```bash
huggingface-cli login
```

### 8.2 Run training

A standard starting point using the ACT policy (good default for manipulation tasks):

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=<HF_USERNAME>/<DATASET_NAME> \
  --output_dir=outputs/train/<your-run-name>
```

Training will checkpoint periodically. Monitor GPU usage with `nvidia-smi`.

**Important:** Multiple teams share the workstation. Be considerate — do not run more than one training job at a time, and check `nvidia-smi` before launching.

### 8.3 (Optional) Push your trained model to Hub

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=<HF_USERNAME>/<DATASET_NAME> \
  --output_dir=outputs/train/<your-run-name> \
  --policy.push_to_hub=true \
  --policy.repo_id=<HF_USERNAME>/<MODEL_NAME>
```

---

## Part 9 — Evaluate the Policy

Back on your laptop, run the trained policy on the physical robot:

```bash
lerobot-eval \
  --policy.path=<HF_USERNAME>/<MODEL_NAME> \
  --robot.type=so100_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=follower_so100 \
  --robot.cameras="{front: {type: opencv, index_or_path: <CAMERA_INDEX>, width: 640, height: 480, fps: 30}}"
```

Watch whether the robot successfully performs the task. If success rate is low, collect more episodes and retrain.

---

## Quick Reference

| Step | Command |
|---|---|
| Find ports | `lerobot-find-port` |
| Find camera | `lerobot-find-cameras` |
| Calibrate leader | `lerobot-calibrate --teleop.type=so100_leader --teleop.port=<PORT> --teleop.id=leader_so100` |
| Calibrate follower | `lerobot-calibrate --robot.type=so100_follower --robot.port=<PORT> --robot.id=follower_so100` |
| Test teleoperation | `lerobot-teleoperate ...` |
| Record dataset | `lerobot-record ...` |
| Train (on workstation) | `lerobot-train --policy.type=act --dataset.repo_id=<HF_USERNAME>/<DATASET>` |
| Evaluate | `lerobot-eval --policy.path=<HF_USERNAME>/<MODEL> ...` |

---

## Troubleshooting

**`lerobot-find-port` detects nothing**
- Make sure the arm is plugged into this machine directly (not a hub you are SSH'd into).
- Try a different USB cable or port.

**`ValueError: Some motors have the same min and max values`**
- A joint was not swept far enough during calibration. Rerun and move that joint through its full range.

**Permission denied on `/dev/tty...` (Linux)**
- Run `sudo usermod -aG dialout $USER` and log out/in.

**Camera not found**
- Run `lerobot-find-cameras` to confirm the index. Try index `1` if `0` doesn't show your camera.

**HF push fails**
- Check your token has Write permission.
- Use the manual upload snippet in Part 7.

**CUDA not available on workstation**
- Run `python -c "import torch; print(torch.cuda.is_available())"` to confirm.
- If False, your PyTorch may not match the CUDA driver. Ask a hackathon organizer for the correct install command.
