import math
import time
import numpy as np
import mujoco #, mujoco_viewer
import mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

import glob
import re
import torch
import onnx
import onnxruntime as ort

# local imports
import argparse
import sys
import os
import numpy
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
import yaml
import glfw

from sim2sim import MujocoSimEnv, InferenceRunner, hydra_mj_config


# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger(__file__)


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--log_length", type=int, default=200, help="Length of the recorded log (in steps).")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--play_demo_traj", action="store_true", default=False, help="play a demo trajectory")
parser.add_argument_group("st_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored.")
parser.add_argument("--logs", type=str, default="/home/thomas/workspace/lumos_ws/st_gym/logs/st_rl/", help="Name of the log folder to resume from.")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
parser.add_argument("--policy_path", type=str, default=None, help="The path of onnx model path of trained policy")
parser.add_argument("--model_path", type=str, default=None, help="Robot XML model path")
parser.add_argument("--export_rknn", action="store_true", default=False, help="export rknn and test it")
parser.add_argument("--saving_data", action="store_true", default=False, help="saving mj sim data")

# append AppLauncher cli args
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
print(f"Hydr args: {hydra_args}")
print(f"logs folder: {args_cli.logs}")

def add_visual_capsule(scene, from_pos, to_pos, radius, color_rgba):
    # Ensure proper dtype and shape
    from_pos = np.asarray(from_pos, dtype=np.float64).reshape(3)
    to_pos = np.asarray(to_pos, dtype=np.float64).reshape(3)

    # Create a new geometry
    #geom = mujoco.MjvGeom()
    scene.ngeom += 1  # increment ngeom
    geom = scene.geoms[scene.ngeom-1]
    mujoco.mjv_initGeom(geom,
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                         np.zeros(3), np.zeros(9), color_rgba.astype(np.float32))
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        from_pos,
        to_pos
    )


def key_call_back( keycode):
    global control_mode, mj_sim_mode, base_velocity
    if chr(keycode) == "R":
        control_mode="RESET"
        logger.info(f"control mode: {control_mode}")
    if chr(keycode) == "K":
        control_mode="STANDUP"
        logger.info(f"control mode: {control_mode}")
    if chr(keycode) == "L":
        control_mode="RL"
        logger.info(f"control mode: {control_mode}")
    if keycode == glfw.KEY_UP:
        base_velocity[0]+=0.1
        logger.info(f"base velocity: {base_velocity}")
    if keycode == glfw.KEY_DOWN:
        base_velocity[0]-=0.1
        logger.info(f"base velocity: {base_velocity}")
    if keycode == glfw.KEY_RIGHT:
        base_velocity[1]-=0.1
        logger.info(f"base velocity: {base_velocity}")
    if keycode == glfw.KEY_LEFT:
        base_velocity[1]+=0.1
        logger.info(f"base velocity: {base_velocity}")
    if chr(keycode) == "Q":
        base_velocity[2]+=0.1
        logger.info(f"base velocity: {base_velocity}")
    if chr(keycode) == "E":
        base_velocity[2] -=0.1
        logger.info(f"base velocity: {base_velocity}")

@hydra_mj_config(args_cli)
def run_mujoco(env_cfg: DictConfig, agent_cfg:DictConfig):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    global control_mode, mj_sim_mode, base_velocity
    if not os.path.isfile(env_cfg.ref_motion.motion_files[0]):
        env_cfg.ref_motion.motion_files = glob.glob(os.path.join(os.getenv("HOME"),env_cfg.ref_motion.motion_files[0][env_cfg.ref_motion.motion_files[0].find("workspace"):]))
    if len(env_cfg.ref_motion.motion_files) < 1:
        #env_cfg.ref_motion.motion_files = glob.glob("/home/thomas/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/dance2_subject4_1871_6771_fps25.pkl")
        env_cfg.ref_motion.motion_files = glob.glob(f"{os.getenv('HOME')}/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/*")
        print(f"The ref motion for training do not exist, change to use {env_cfg.ref_motion.motion_files}")

    #env_cfg.ref_motion.motion_files = glob.glob(f"{os.getenv('HOME')}/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/Mma_Kick_fps30.pkl")
    logger.info(f"Ref motion path: {env_cfg.ref_motion.motion_files}")

    env_cfg.ref_motion.frame_begin = None #0 #175
    env_cfg.ref_motion.frame_end = None #None #2650
    env_cfg.ref_motion.ref_length_s= None #12.1+4
    env_cfg.ref_motion.random_start = False
    env_cfg.ref_motion.random_start=False
    specify_init_values = {}
    specify_init_values["root_rot_x"] = 0
    specify_init_values["root_rot_y"] = 0
    specify_init_values["root_rot_z"] = 0
    specify_init_values["root_rot_w"] = 1
    specify_init_values["root_pos_z"] = 0.86
    specify_init_values["left_hip_pitch_joint_dof_pos"] = -0.37
    specify_init_values["right_hip_pitch_joint_dof_pos"] = -0.37
    specify_init_values["left_knee_joint_dof_pos"] = 0.74
    specify_init_values["right_knee_joint_dof_pos"] = 0.74
    specify_init_values["left_ankle_pitch_joint_dof_pos"] = -0.37
    specify_init_values["right_ankle_pitch_joint_dof_pos"] = -0.37
    specify_init_values["left_shoulder_roll_joint_dof_pos"] = 0.25
    specify_init_values["right_shoulder_roll_joint_dof_pos"] = -0.25
    specify_init_values["left_elbow_joint_dof_pos"] = 1.2
    specify_init_values["right_elbow_joint_dof_pos"] = 1.2
    env_cfg.ref_motion.specify_init_values = None #specify_init_values #if env_cfg.ref_motion.specify_init_values is not None else None
    env = MujocoSimEnv(env_cfg, args_cli)

    control_mode="STANDUP"
    base_velocity= [0,0,0]
    runner=InferenceRunner(env, agent_cfg, args_cli)
    policy = runner.get_inference_policy()
    actions=torch.zeros(env.num_env,env.joint_num).to(env.device)

    velocity_commands = np.array([0,0,0])


    # key body index
    robot_bodies = ["left_elbow_link", "right_elbow_link","left_hip_pitch_link","right_hip_pitch_link","left_shoulder_roll_link","right_shoulder_roll_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link"] #"left_wrist_roll_link","right_wrist_roll_link"]
    body_index = [[env.ref_motion.trajectory_fields.index(key + kk) for kk in ["_pos_x_w", "_pos_y_w", "_pos_z_w"]] for key in robot_bodies]
    print(f" body index : {body_index}")


    with mujoco.viewer.launch_passive(env.mj_model, env.mj_data, key_callback=key_call_back, show_right_ui=False, show_left_ui=False) as viewer:
        cam = viewer.cam
        cam.distance = 4.0 ;cam.azimuth = 135; cam.elevation = -10; cam.lookat = [0,0,0]
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid=1;

        # adding capsule gemos for vis traj tracking
        add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.04, np.array([1, 0, 0, 1]))
        for _ in range(10):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.04, np.array([0, 1, 0, 1]))

        while viewer.is_running():
            cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING
            step_start = time.time()
            # Obtain an observation
            env.set_commands(base_velocity)
            obs, extras = env.step(actions)
            if control_mode=="RL":
                actions = policy(obs) # update policy with higher frq, but use low ref frq
                if args_cli.saving_data:
                    env.update_log(actions, obs, extras)
                if(env.ref_motion.frame_idx==env.ref_motion.clip_frame_num-1):
                    logger.info(f"✅ Done, frame idx is {env.ref_motion.frame_idx}")
                    if not args_cli.saving_data:
                        env.reset()
                        runner.reset()
                    else:
                        break
            elif control_mode=="STANDUP":
                actions=torch.zeros(env.num_env, env.joint_num).to(env.device)
                #env.reset()
            elif control_mode=="RESET":
                actions=torch.zeros(env.num_env, env.joint_num).to(env.device)
                env.reset()
                runner.reset()
                if env.ref_motion.frame_idx>1:
                    logger.info(f"Reset, frame idx is {env.ref_motion.frame_idx}")
            else:
                raise f"unkown control mode"

            # visualizing robot joints
            if robot_bodies is not None:
                for i,idx in enumerate(body_index):
                    viewer.user_scn.geoms[1+i].pos = env.ref_motion.data[:, idx]
                
            viewer.sync()
            time_until_next_step = env.step_dt - (time.time() - step_start)
            viewer.user_scn.geoms[0].pos = (0,0,0)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        # save log
        if args_cli.saving_data:
            env.save_log(os.path.dirname(runner.policy_path))
        if args_cli.export_rknn:
            # test rknn  and save testing results
            store_rknn_action = runner.test_rknn(env.store_obs, env.store_action)

            # test rknn AGAIN with data from file and save testing results
            #store_rknn_action = runner.test_rknn(test_obs_data_path=mj_onnx_obs_path, test_action_data_path=mj_onnx_action_path)
            #mj_rknn_action_path = os.path.join(eval_result_folder, "store_mj_rknn_action.txt")
            #np.savetxt(mj_rknn_action_path, store_rknn_action, fmt="%.4f")
            logger.info(f"Successfully export rknn and test it in {eval_result_folder}")


if __name__=="__main__":
    run_mujoco()
