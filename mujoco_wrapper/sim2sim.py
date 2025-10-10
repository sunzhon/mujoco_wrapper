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

# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger(__file__)


import functools
from collections.abc import Callable

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")


def replace_slices_with_strings(data: dict) -> dict:
    """Replace slice objects with their string representations in a dictionary.

    Args:
        data: The dictionary to process.

    Returns:
        The dictionary with slice objects replaced by their string representations.
    """
    if isinstance(data, dict):
        return {k: replace_slices_with_strings(v) for k, v in data.items()}
    elif isinstance(data, slice):
        return f"slice({data.start},{data.stop},{data.step})"
    else:
        return data



def hydra_mj_config(args_cli) -> Callable:
    """Decorator to handle the Hydra configuration for a load_run.

    This decorator registers the task to Hydra and updates the environment and agent configurations from Hydra parsed
    command line arguments.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the envrionment's and agent's configurations updated from command line arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # define the new Hydra main function
            config_path=os.path.abspath(os.path.join(args_cli.logs, args_cli.experiment_name, args_cli.load_run, "params"))
            print("[Hydr Info] parse env cfg and agent cfg ...")
            import yaml
            def hydra_main(config_path):
                with open(os.path.join(config_path, "env.yaml")) as f:
                    env_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)  # o
                    env_cfg = replace_slices_with_strings(env_cfg)
                    #env_cfg['scene']['robot']['default_joint_limits']= env_cfg['scene']['robot']['default_joint_limits'].tolist()
                    env_cfg = OmegaConf.create(env_cfg)

                with open(os.path.join(config_path, "agent.yaml")) as f:
                    agent_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)  # o
                    agent_cfg = replace_slices_with_strings(agent_cfg)
                    agent_cfg = OmegaConf.create(agent_cfg)

                    # specify directory for logging experiments
                    log_root_path = os.path.join(args_cli.logs, agent_cfg.experiment_name)
                    log_root_path = os.path.abspath(log_root_path)
                    print(f"[INFO] Loading experiment from directory: {log_root_path}")
                    #agent_cfg.resume_path = os.path.join(log_root_path, env_cfg.load_run, agent_cfg.load_checkpoint)
                    agent_cfg.onnx_path = os.path.join(log_root_path, args_cli.load_run,"exported","policy.onnx")
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main(config_path)
        return wrapper
    return decorator




class MujocoSimEnv:

    def __init__(self, env_cfg: DictConfig, args_cli):
        """Play with RSL-RL agent."""
        self.cfg = env_cfg

        #1)  loading mujoco model and data
        if args_cli.model_path is None:
            robot_model_path = env_cfg.scene.robot.spawn.usd_path
            robot_model_dirname = os.path.dirname(os.path.dirname(robot_model_path))
            robot_model_name = os.path.basename(robot_model_path).split(".")[0]
            self.robot_model_xml_path = os.path.join(robot_model_dirname,"mjcf",robot_model_name+".xml")
        else:
            self.robot_model_xml_path = os.path.join(args_cli.model_path)

        if not os.path.isfile(self.robot_model_xml_path):
            self.robot_model_xml_path = os.path.join(os.getenv("HOME"),self.robot_model_xml_path[self.robot_model_xml_path.find("workspace"):])
        logger.info(f"XML model path: {self.robot_model_xml_path}")

        self.mj_model = mujoco.MjModel.from_xml_path(self.robot_model_xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mj_model.opt.timestep = env_cfg.sim.dt
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.step_dt = env_cfg.sim.dt *self.cfg.decimation

        self.device="cpu"
        self.step_counter = 0
        self.num_env = 1
        from collections import deque
        # simulate sensory delay
        self.obs_buf=deque(maxlen=1)

        # loading joint names from mujoco
        self.mujoco_joint_names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.mj_model.njnt)][1:]
        self.policy_joint_names = env_cfg.scene.robot.joint_names
        self.policy_dof_index = [self.mujoco_joint_names.index(key) for key in self.policy_joint_names]
        self.mujoco_dof_index = [self.policy_joint_names.index(key) for key in self.mujoco_joint_names]
        logger.info(f"mujoco joint names: {self.mujoco_joint_names}")
        logger.info(f"policy joint names: {self.policy_joint_names}")

        self.joint_num = len(self.mujoco_joint_names)
        logger.info(f"mujoco joint number: {self.joint_num}")


        # Identifiers for the floor, right foot, and left foot
        self.floor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.right_foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.left_foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        logger.info(f" floor id: {self.floor_id}, right foot id: {self.right_foot_id}, left feet id: {self.left_foot_id}")



        #4) prepare variables 
        # get defualt joint pos
        self.default_dof_pos = np.zeros(env_cfg.num_actions)
        default_dof_pos_patterns = env_cfg.scene.robot.init_state.joint_pos
        for key, value in default_dof_pos_patterns.items():
            pattern = re.compile(key)
            match_fields = [s for s in self.mujoco_joint_names if pattern.match(s)]
            self.default_dof_pos[[self.mujoco_joint_names.index(key) for key in match_fields]] = value

        #5) get joint stiffness, damping, and effort limits
        actuators_cfg = env_cfg.scene.robot.actuators
        self.kps, self.kds = torch.zeros(self.num_env, env_cfg.num_actions), torch.zeros(self.num_env, env_cfg.num_actions)
        self.effort_limit_sim = torch.zeros(self.num_env, env_cfg.num_actions)
        self.velocity_limit_sim = torch.zeros(self.num_env, env_cfg.num_actions)
        for joint_group_name, cfg in actuators_cfg.items():
            for key, value in cfg.stiffness.items():
                pattern = re.compile(key)
                match_fields = [s for s in self.mujoco_joint_names if pattern.match(s)]
                self.kps[:,[self.mujoco_joint_names.index(key) for key in match_fields]] = value
            for key, value in cfg.damping.items():
                pattern = re.compile(key)
                match_fields = [s for s in self.mujoco_joint_names if pattern.match(s)]
                self.kds[:,[self.mujoco_joint_names.index(key) for key in match_fields]] = value
                self.effort_limit_sim[:,[self.mujoco_joint_names.index(key) for key in match_fields]]=cfg.effort_limit_sim
                self.velocity_limit_sim[:,[self.mujoco_joint_names.index(key) for key in match_fields]]=cfg.velocity_limit_sim

        self.kps_bk = self.kps.clone().detach()
        self.kds_bk = self.kds.clone().detach()
        logger.info(f"joint control stiffness: {self.kps}")
        logger.info(f"joint control damping: {self.kds}")

        logger.info(f"[Sim2Sim] default joint pos: {self.default_dof_pos}")

        self.obs_list=[]
        for key, value in self.cfg.observations.policy.items():
            if isinstance(value, object):
                if hasattr(value, "func"):
                    self.obs_list.append(key)
        logger.info(f"[Sim2Sim] observation components are {self.obs_list}")


        self.last_actions=torch.zeros(self.num_env, env_cfg.num_actions).to(self.device)

        # init base velocity
        self.base_velocity = torch.zeros(self.num_env, 3, device=self.device)

        # loading ref motion data
        self.store_obs = []
        self.store_target_q = []
        self.store_action = []
        self.store_ref_motion = []
        self.store_extras = []
        
        # loading ref motion
        self.load_refmotion()


    def load_refmotion(self):
        """
        GET CFG
        """
        if not hasattr(self.cfg, "ref_motion"):
            raise Exception("Did not have ref motion cfg")

        ref_motion_cfg = self.cfg.ref_motion
        #) loading amp if needed
        #from legged_robots.data_manager.motion_loader import RefMotionLoader
        from refmotion_manager.motion_loader import RefMotionLoader
        #tyle_fields = ref_motion_cfg.init_state_fields
        #ref_motion_cfg.style_fields = style_fields
        ref_motion_cfg.trajectory_num = 1
        #ref_motion_cfg.amp_obs_history_length = 1 # amp_obs_len
        #ref_motion_cfg.frame_begin=10
        ref_motion_cfg.random_start = False
        ref_motion_cfg.device = self.device
        
        # padding key was not added
        if ref_motion_cfg.specify_init_values is not None:
            for k1 in self.mujoco_joint_names:
                key = k1+"_dof_pos"
                if key not in ref_motion_cfg.specify_init_values.keys():
                    ref_motion_cfg.specify_init_values[key] = 0.0

        # loading ref motion
        self.ref_motion = RefMotionLoader(ref_motion_cfg)

        # get init state of root and dof
        init_dof_pos_index = [self.ref_motion.trajectory_fields.index(key) for key in [f+"_dof_pos" for f in self.mujoco_joint_names]]
        self.init_joint_pos = self.ref_motion.preloaded_s[:,0,init_dof_pos_index]

        init_dof_vel_index = [self.ref_motion.trajectory_fields.index(key) for key in [f+"_dof_vel" for f in self.mujoco_joint_names]]
        self.init_joint_vel = self.ref_motion.preloaded_s[:,0,init_dof_vel_index]
        
        root_pos_fields = ['root_pos_x', 'root_pos_y', 'root_pos_z', 'root_rot_w', 'root_rot_x', 'root_rot_y', 'root_rot_z']
        init_root_pos_index = [self.ref_motion.trajectory_fields.index(key) for key in root_pos_fields]
        self.init_root_pos = self.ref_motion.preloaded_s[:,0,init_root_pos_index]

    def set_commands(self, velocity_commands):
        self.base_velocity[:,0] =  velocity_commands[0]
        self.base_velocity[:,1] =  velocity_commands[1]
        self.base_velocity[:,2] =  velocity_commands[2]


    def get_obs(self):
        '''Extracts an observation from the mujoco data structure
        '''
        
        #0) get mj data from robot states and sensors
        data = self.mj_data
    
        #1) get sensor data of base/root link
        base_pos_w =  torch.tensor(data.sensor('position').data.astype(np.float32)).reshape(self.num_env,-1)
        base_quat_w =  torch.tensor(data.sensor('orientation').data.astype(np.float32)).reshape(self.num_env,-1)
        base_lin_vel_b = torch.tensor(data.sensor('linear-velocity').data.astype(np.float32).reshape(self.num_env,-1), dtype=torch.float32, device=self.device)
        base_ang_vel_b = torch.tensor(data.sensor('angular-velocity').data.astype(np.float32).reshape(self.num_env,-1),dtype=torch.float32,device=self.device)
        
        #2) calculate gvec
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.float32)
        r = R.from_quat(quat)
        v = r.apply(data.qvel[:3], inverse=True).astype(np.float32)  # In the base frame
        projected_gravity = torch.tensor(r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.float32).reshape(self.num_env,-1),dtype=torch.float32,device=self.device)

        #3) get joint pos and vel
        q = data.qpos.astype(np.float32)[7:]
        dq = data.qvel.astype(np.float32)[6:]
        joint_pos = torch.from_numpy(q[self.policy_dof_index].reshape(self.num_env,-1)).to(self.device)
        joint_vel = torch.from_numpy(dq[self.policy_dof_index].reshape(self.num_env,-1)).to(self.device)
        p_joint_pos = joint_pos
        p_joint_vel = joint_vel

        
        #3.1) get joint torques 
        #joint_tor = torch.from_numpy(data.actuator_force.copy()) # actuator output forces
        joint_tor = torch.from_numpy(data.qfrc_actuator.copy())[6:] # remvoe float joint, joint force get from actuator via transmission


        #3.2) get feet contact forces
        grf = self.get_grf()
        #logger.info(f"l_grf: {grf[0]}, r_grf: {grf[1]}")


        #4) get keypoint/joints frame location and linear velocity
        body_lin_vel_w=[]
        body_pos_w=[]
        expressive_link_name = getattr(self.cfg.ref_motion,"expressive_link_name", None)
        vel = np.zeros(6)
        if expressive_link_name is not None:
            for name in expressive_link_name:
                body_id = self.mj_model.body(name).id
                mujoco.mj_objectVelocity(self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
                body_lin_vel_w.append(torch.tensor(vel[3:], device=self.device, dtype=torch.float32))
                body_pos_w.append(torch.tensor(self.mj_data.xpos[body_id], device=self.device, dtype=torch.float32))  # (3,)
            
            body_lin_vel_w = torch.stack(body_lin_vel_w)
            body_pos_w = torch.stack(body_pos_w)

            # calculate pos and vel in root frame
            from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat
            body_pos_b = body_pos_w - base_pos_w         # (envs, bodies, 3)

            body_pos_b = quat_apply_inverse(base_quat_w.expand(body_pos_b.shape[0], -1), body_pos_b)
            body_lin_vel_b = quat_apply_inverse(base_quat_w.expand(body_lin_vel_w.shape[0],-1), body_lin_vel_w)

            body_pos_w = body_pos_w.reshape(self.num_env,-1)
            body_lin_vel_w = body_lin_vel_w.reshape(self.num_env,-1)  # shape: (num_links, 3)
            body_pos_b = body_pos_b.reshape(self.num_env, -1)
            body_lin_vel_b = body_lin_vel_b.reshape(self.num_env, -1)

        #5) goal status
        ref_fields={}
        if hasattr(self,"ref_motion"):
            velocity_commands = self.base_velocity + self.ref_motion.base_velocity_b
            if "velocity_commands" not in ref_fields:
                ref_fields["velocity_commands"]=["base_vel_cmd_x", "base_vel_cmd_y", "base_vel_cmd_z"]
            # ref motions
            if hasattr(self.ref_motion, "style_goal"):
                style_goal_commands = self.ref_motion.style_goal
                if "style_goal_commands" not in ref_fields:
                    ref_fields["style_goal_commands"] = self.cfg.ref_motion.style_goal_fields
            if hasattr(self.ref_motion, "expressive_goal"):
                expressive_goal_commands = self.ref_motion.expressive_goal
                if "expressive_goal_commands" not in ref_fields:
                    ref_fields["expressive_goal_commands"] = self.cfg.ref_motion.expressive_goal_fields
        else:
            velocity_commands = self.base_velocity

        last_actions = self.last_actions
        p_last_actions = last_actions

        #6) choose necessary obs and critic_obs terms
        obs_list=[]
        for key, value in self.cfg.observations.policy.items():
            if hasattr(value, "func"):
                if value.func is not None:
                    obs_list.append(locals()[key])
        obs = torch.cat(obs_list, dim=1)

        critic_obs_list = []
        if hasattr(self.cfg.observations,"critic"):
            for key, value in self.cfg.observations.critic.items():
                if hasattr(value, "func"):
                    if value.func is not None:
                        if key in locals(): # NOTE, only adding existing critic obs term in sim2sim
                            critic_obs_list.append(locals()[key])
            critic_obs = torch.cat(critic_obs_list, dim=1)
        else:
            critic_obs = obs

        # store ref_motion frames for real deploy
        ref_motion_list = []
        self.ref_motion_fields = []
        if hasattr(self.cfg,"ref_motion"):
            for key, value in self.cfg.observations.policy.items():
                if hasattr(value, "func"):
                    if value.func is not None:
                        if key in ["velocity_commands", "style_goal_commands","expressive_goal_commands"]:
                            ref_motion_list.append(locals()[key])
                            self.ref_motion_fields.extend(ref_fields[key])
            ref_motion = torch.cat(ref_motion_list, dim=1)
        else:
            ref_motion = None

        self.obs_buf.append(obs)
        extras = {"ref_motion": ref_motion,"critic_obs": critic_obs, "joint_pos": np.squeeze(joint_pos), "joint_vel": np.squeeze(joint_vel), "joint_tor": joint_tor, "grf": grf}
        
        return self.obs_buf[0], extras
    
    
    def pd_control(self, target_q):
        '''Calculates torques from position commands
        '''

        self.q = torch.tensor(self.mj_data.qpos.astype(np.double)[7:],device=self.device).reshape(self.num_env,-1)
        self.dq = torch.tensor(self.mj_data.qvel.astype(np.double)[6:],device=self.device).reshape(self.num_env,-1)

        target_dq = torch.tensor(np.zeros((self.cfg.num_actions), dtype=np.double),device=self.device).reshape(self.num_env,-1)

        return (target_q - self.q) * self.kps + (target_dq - self.dq) * self.kds
    

    def step(self, actions):

        # apply actions
        self.last_actions = actions.clone()
        mujoco_actions = actions[:, self.mujoco_dof_index].to(dtype=torch.float32, device=self.device)


        if self.cfg.actions.joint_pos.offset is not None:
            mujoco_actions +=self.cfg.actions.joint_pos.offset
        if self.cfg.actions.joint_pos.clip is not None:
            mujoco_actions = np.clip(mujoco_actions, -self.cfg.actions.joint_pos.clip, self.cfg.actions.joint_pos.clip)

        # update
        for _ in range(self.cfg.decimation):
            target_q = mujoco_actions * self.cfg.actions.joint_pos.scale  + self.default_dof_pos
            tau = self.pd_control(target_q)  # Calc torques
            self.mj_data.ctrl = tau #*0.0 +100
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.target_q = target_q

        # updating ref motion
        if hasattr(self,"ref_motion"):
            if(self.ref_motion.frame_idx==self.ref_motion.clip_frame_num-1):
                self.ref_motion.reset()
                logger.info(f"📌 Reset ref motion, Loading ref motion from begin")
            self.ref_motion.step()

        self.step_counter+=1

        # get obs 
        obs, extras = self.get_obs() 
        
        self.log = (actions,obs, extras)

        return obs, extras


    def reset(self):

        # 1. Reset mjData to match initial conditions from XML
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        
        # 2. (Optional) Reset simulation time
        self.mj_data.time = 0.0
        self.step_counter = 0
        self.kps = self.kps_bk.clone()

        # updating ref motion
        if hasattr(self,"ref_motion"):
            self.ref_motion.reset()

        # 3. (Optional) Set initial joint positions or velocities
        if hasattr(self, "init_root_pos"):
            self.mj_data.qpos[:7] = self.init_root_pos
        if hasattr(self, "init_joint_pos"):
            self.mj_data.qpos[7:] = self.init_joint_pos

        if hasattr(self, "init_joint_vel"):
            self.mj_data.qvel[6:] = self.init_joint_vel

        # 4. Re-forward kinematics to make data consistent
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 5. empty store
        self.store_obs = []
        self.store_action = []
        self.store_target_q = []
        self.store_ref_motion = []
        self.store_extras = []

        return 0

    def update_log(self, actions, obs, extras):
        
        self.store_obs.append(obs)
        self.store_target_q.append(self.target_q)
        self.store_action.append(actions)
        self.store_ref_motion.append(extras["ref_motion"])
        self.store_extras.append(extras)


    def save_log(self, saving_folder):
        logger.info(f"Saving data ... ")
        # array obs, action, and ref motion
        store_action = torch.cat(self.store_action).cpu().numpy()
        store_target_q = torch.cat(self.store_target_q).cpu().numpy()
        store_obs = torch.cat(self.store_obs).cpu().numpy()
        store_ref_motion = torch.cat(self.store_ref_motion).cpu().numpy()
        store_extras = self.store_extras

        # get data saving folder
        eval_result_folder = saving_folder

        # saving joint names of policy and robots
        joint_name_path=os.path.join(eval_result_folder, "joint_names.yaml")
        with open(joint_name_path, "w") as file:
            yaml.dump({"policy_joint_names":list(self.policy_joint_names),"robot_joint_names":self.mujoco_joint_names}, file, default_flow_style=False)


        # saving fileds of ref motion
        ref_motion_fields_path=os.path.join(eval_result_folder, "ref_motion_fields.yaml")
        with open(ref_motion_fields_path, "w") as file:
            yaml.dump({"ref_motion_fields":self.ref_motion_fields}, file, default_flow_style=False)

        # saving kp and kds
        kpkds_path=os.path.join(eval_result_folder, "kp_kd.yaml")
        with open(kpkds_path, "w") as file:
            yaml.dump({"kps":self.kps.squeeze().numpy().tolist(),"kds": self.kds.squeeze().numpy().tolist(), 
                "scales":self.cfg.num_actions*[self.cfg.actions.joint_pos.scale]}, file, default_flow_style=False)

        # saving obs, action, and ref motion
        mj_onnx_action_path = os.path.join(eval_result_folder, "store_mj_onnx_action.txt")
        np.savetxt(mj_onnx_action_path, store_action, fmt="%.4f")

        mj_onnx_target_q_path = os.path.join(eval_result_folder, "store_mj_onnx_target_q.txt")
        np.savetxt(mj_onnx_target_q_path, store_target_q, fmt="%.4f")

        mj_onnx_obs_path = os.path.join(eval_result_folder, "store_mj_onnx_obs.txt")
        np.savetxt(mj_onnx_obs_path, store_obs, fmt="%.4f")

        mj_ref_motion_path = os.path.join(eval_result_folder, "store_ref_motion.txt")
        np.savetxt(mj_ref_motion_path, store_ref_motion, fmt="%.4f")

        mj_ref_init_pos_path = os.path.join(eval_result_folder, "store_ref_init_dof_pos.txt")
        np.savetxt(mj_ref_init_pos_path, self.init_joint_pos, fmt="%.4f")

        mj_extras_path = os.path.join(eval_result_folder, "store_extras.pkl")
        import joblib
        joblib.dump(store_extras, mj_extras_path)    # Save

        logger.info(f"Successfully saving mj data  with shape {store_ref_motion.shape}: in {eval_result_folder}")



    def get_grf(self):
        r_grf = np.zeros(6, dtype=float)
        l_grf = np.zeros(6, dtype=float)

        # Iterate over contacts
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2
            # Right foot contact with floor
            if self.floor_id in {geom1, geom2}:
                if self.right_foot_id in {self.mj_model.geom_bodyid[geom1], self.mj_model.geom_bodyid[geom2]}:
                    tmp_force = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(self.mj_model, self.mj_data, i, tmp_force)
                    r_grf += tmp_force
                # Left foot contact with floor
                elif self.left_foot_id in {self.mj_model.geom_bodyid[geom1], self.mj_model.geom_bodyid[geom2]}:
                    tmp_force = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(self.mj_model, self.mj_data, i, tmp_force)
                    l_grf += tmp_force

        return (l_grf, r_grf)


class InferenceRunner:

    def __init__(self, env: MujocoSimEnv, agent_cfg, args_cli):

        # loading policy from onnx and checkin it
        self.cfg = agent_cfg
        if args_cli.policy_path is not None:
            self.policy_path = args_cli.policy_path
        else:
            self.policy_path = agent_cfg.onnx_path
        logger.info(f"policy path: {self.policy_path}")
        # checking model
        onnx_model = onnx.load(self.policy_path)
        logger.info(f"checking policy: {onnx.checker.check_model(onnx_model)}")

        #3.1) 创建 ONNX 运行时会话
        self.ort_session = ort.InferenceSession(agent_cfg.onnx_path)

        #3.2) 获取输入和输出信息
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape
        self.output_name = self.ort_session.get_outputs()[0].name
        self.output_shape = self.ort_session.get_outputs()[0].shape

        if "Recurrent" in self.cfg.policy.class_name:
            self.is_recurrent=True
            if self.cfg.policy.rnn_type=="lstm":
                self.h_input_name = self.ort_session.get_inputs()[1].name
                self.c_input_name = self.ort_session.get_inputs()[2].name
                self.h_output_name = self.ort_session.get_outputs()[1].name
                self.c_output_name = self.ort_session.get_outputs()[2].name
                self.hidden_state = np.zeros((1,1,256)).astype(np.float32)
                self.cell_state = np.zeros((1,1,256)).astype(np.float32)
                self.rnn_type = "lstm"
            elif self.cfg.policy.rnn_type=="gru":
                self.h_input_name = self.ort_session.get_inputs()[1].name
                self.h_output_name = self.ort_session.get_outputs()[1].name
                self.hidden_state = np.zeros((1,1,256)).astype(np.float32)
                self.rnn_type = "gru"
        else:
            self.is_recurrent=False

        # 准备输入数据
        # input_data = np.ones(input_shape).astype(np.float32)

        self.export_rknn_model_flag = False

    def reset(self):
        if "Recurrent" in self.cfg.policy.class_name:
            self.is_recurrent=True
            if self.cfg.policy.rnn_type=="lstm":
                self.h_input_name = self.ort_session.get_inputs()[1].name
                self.c_input_name = self.ort_session.get_inputs()[2].name
                self.h_output_name = self.ort_session.get_outputs()[1].name
                self.c_output_name = self.ort_session.get_outputs()[2].name
                self.hidden_state = np.zeros((1,1,256)).astype(np.float32)
                self.cell_state = np.zeros((1,1,256)).astype(np.float32)
            elif self.cfg.policy.rnn_type=="gru":
                self.h_input_name = self.ort_session.get_inputs()[1].name
                self.h_output_name = self.ort_session.get_outputs()[1].name
                self.hidden_state = np.zeros((1,1,256)).astype(np.float32)
        else:
            self.is_recurrent=False


    def get_inference_policy(self):
        def policy(obs):
            if not self.is_recurrent:
                if isinstance(obs, torch.Tensor):
                    obs = obs.detach().cpu().numpy().astype(np.float32)
                elif isinstance(obs, np.ndarray):
                    obs = obs.astype(np.float32)
                else:
                    raise TypeError("Expected input to be torch.Tensor or np.ndarray")
                actions = self.ort_session.run([self.output_name], {self.input_name: obs})[0]
                actions = torch.from_numpy(actions)

                return actions
            else:
                if isinstance(obs, torch.Tensor):
                    obs = obs.detach().cpu().numpy().astype(np.float32)
                elif isinstance(obs, np.ndarray):
                    obs = obs.astype(np.float32)
                else:
                    raise TypeError("Expected input to be torch.Tensor or np.ndarray")
                if self.rnn_type=="lstm":
                    outputs = self.ort_session.run([self.output_name, self.h_output_name, self.c_output_name], {self.input_name: obs, self.h_input_name: self.hidden_state,self.c_input_name: self.cell_state})
                    self.hidden_state=outputs[1]
                    self.cell_state =outputs[2]
                elif self.rnn_type=="gru":
                    outputs = self.ort_session.run([self.output_name, self.h_output_name], {self.input_name: obs, self.h_input_name: self.hidden_state})
                    self.hidden_state=outputs[1]

                output_data = outputs[0]
                actions = output_data[0]
                actions = torch.from_numpy(actions).reshape(1,-1)
                return actions

        return policy

            


    def export_rknn(self, test_obs, test_actions):
        
        if not self.export_rknn_model_flag:
            from rknn.api import RKNN
            """ if no rknnpackage, install it by: pip install rknn-toolkit2"""

            self.rknn_model_path = self.policy_path.replace('.onnx','.rknn')
            DATASET = './dataset.txt'
            # Create RKNN object
            rknn = RKNN(verbose=True)
            # pre-process config
            logger.info('--> Config model')
            rknn.config(mean_values=[[0]*self.input_shape[1]], std_values=[[1]*self.input_shape[1]], target_platform='rk3588')
            logger.info('configing model  done')

            # Load ONNX model
            logger.info(f"--> Loading model: {self.policy_path}")
            ret = rknn.load_onnx(model=self.policy_path)
            if ret != 0:
                logger.info('Load model failed!')
                exit(ret)
            logger.info('loading model done')

            # Build model
            logger.info('--> Building model')
            ret = rknn.build(do_quantization=False, dataset=DATASET)
            if ret != 0:
                logger.info('Build model failed!')
                exit(ret)
            logger.info('done')

            # Export RKNN model
            logger.info('--> Export rknn model')
            ret = rknn.export_rknn(self.rknn_model_path)
            if ret != 0:
                logger.info('Export rknn model failed!')
                exit(ret)
            logger.info('done')

            # Init runtime environment
            logger.info('--> Init runtime environment')
            ret = rknn.init_runtime()
            if ret != 0:
                logger.info('Init runtime environment failed!')
                exit(ret)
            logger.info('done')

            # Inference
            logger.info('--> Running model')
            rknn_actions = rknn.inference(inputs=[test_obs])
            logger.info(f"test obs\n: {[ss for ss in test_obs]}")
            logger.info(f"rknn actions\n: {[ss for ss in rknn_actions[0]]}")
            logger.info(f"test( onnx) actions\n: {[ss for ss in test_actions[0]]}")

            rmse = np.sqrt(np.mean((test_actions - rknn_actions[0]) ** 2))
            logger.info(f"RKNN Action RMSE: {rmse}")

            rknn.release()
            self.export_rknn_model_flag = True

    def test_rknn(self, test_obs=None, test_action=None, test_obs_data_path=None, test_action_data_path=None):

        # Create RKNN object
        from rknn.api import RKNN
        self.rknn_model_path = self.policy_path.replace('.onnx','.rknn')
        DATASET = './dataset.txt'
        # Create RKNN object
        rknn = RKNN(verbose=True)
        # pre-process config
        logger.info('--> Config model')
        rknn.config(mean_values=[[0]*self.input_shape[1]], std_values=[[1]*self.input_shape[1]], target_platform='rk3588')
        logger.info('configing model  done')

        logger.info(f"--> Loading model: {self.policy_path}")
        ret = rknn.load_onnx(model=self.policy_path)
        if ret != 0:
            logger.info('Load model failed!')
            exit(ret)
        logger.info('loading model done')

        # Build model
        logger.info('--> Building model')
        ret = rknn.build(do_quantization=False, dataset=DATASET)
        if ret != 0:
            logger.info('Build model failed!')
            exit(ret)
        logger.info('done')

        # Export RKNN model
        logger.info('--> Export rknn model')
        ret = rknn.export_rknn(self.rknn_model_path)
        if ret != 0:
            logger.info('Export rknn model failed!')
            exit(ret)
        logger.info('done')

        # Init runtime environment
        ret = rknn.init_runtime()
        if ret != 0:
            logger.info('Init runtime environment failed!')
            exit(ret)
        logger.info('done')

        if test_obs_data_path is not None:
            test_obs = np.loadtxt(test_obs_data_path, delimiter=" ")
        if test_action_data_path is not None:
            test_action = np.loadtxt(test_action_data_path, delimiter=" ")

        if test_obs is None:
            test_obs = np.zeros(1,self.input_shape)
        if test_action is None:
            test_action = np.zeros(1,self.output_shape)

        test_obs = test_obs.astype(np.float32)
        test_action = test_action.astype(np.float32)

        logger.info(f"test action shape: {test_action.shape}")
        logger.info(f"test obs shape: {test_obs.shape}")

        # Inference
        rknn_actions = []
        for idx in range(min(test_obs.shape[0],test_action.shape[0])):
            rknn_actions.append(rknn.inference(inputs=[test_obs[idx,:].reshape(1,-1)])[0])

        rknn_action = np.array(rknn_actions)[:,0,:]

        # Evaluation
        rmse = np.sqrt(np.mean((test_action - rknn_action[0]) ** 2))
        logger.info(f"RKNN Action RMSE: {rmse}")

        mj_rknn_action_path = os.path.join(os.path.dirname(self.rknn_model_path), "store_mj_rknn_action.txt")
        np.savetxt(mj_rknn_action_path, rknn_action, fmt="%.4f")

        return rknn_action

