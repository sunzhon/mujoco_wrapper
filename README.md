# Mujoco Wrapper

# create virtual env
pip install onnx onnxruntime opencv-python 
pip install rknn-toolkit2==2.2.1 
pip install mujoco hydra-core

# macOS
Use the `macos` branch on Mac. The MuJoCo passive viewer needs the macOS GUI event loop to be initialized before running `sim2mujoco.py`.

```bash
conda activate env_isaaclab
pip install -e .
pip install mujoco hydra-core onnx onnxruntime opencv-python glfw
mjlab
```

Keep `mjlab` running, then start sim2mujoco from another terminal:

```bash
cd ~/workspace/lumos_ws/st_gym/third_party/mujoco_wrapper
python mujoco_wrapper/sim2mujoco.py
```
