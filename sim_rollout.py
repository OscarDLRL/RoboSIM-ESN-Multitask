"""Physical continuation of ManiSkill PickCube/PushCube from recorded state K.

Two modes:
  oracle: original actions after K, checks the simulator/controller baseline.
  esn:    predicted qpos -> pd_joint_pos actions after K, open-loop.

Uses original env_states AT K to initialize identical scenes, not future env_states.
Never uses saved future cube positions to manipulate physics.
Requires original demonstrations' control_mode == pd_joint_pos.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401 register tasks


def to_python(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    v = np.asarray(v)
    return bool(v.reshape(-1)[0]) if v.size == 1 else bool(v.any())


def get_state_at(group, timestep, device):
    """Convert a HDF5 dictionary-of-state-sequences to a ManiSkill state dict."""
    def rec(node):
        if isinstance(node, h5py.Dataset):
            return torch.as_tensor(np.asarray(node[timestep]), device=device).unsqueeze(0)
        return {name: rec(child) for name, child in node.items()}
    return rec(group['env_states'])


def default_controller(env):
    controllers = env.unwrapped.agent.controller.controllers
    if 'arm' not in controllers or 'gripper' not in controllers:
        raise RuntimeError('Expected composite Panda controller with arm and gripper')
    return controllers['arm'], controllers['gripper']


def to_action(qpos, env):
    """Convert physical Panda qpos to configured joint position control actions."""
    q = np.asarray(qpos, np.float32).reshape(-1)
    if len(q) != 9 or not np.isfinite(q).all():
        raise ValueError('Expected 9 finite joint positions')
    arm, grip = default_controller(env)
    arm_cfg = arm.config
    if bool(getattr(arm_cfg, 'use_delta', False)):
        raise RuntimeError('Only absolute pd_joint_pos controller is supported')
    arm_target = q[:7].copy()
    if bool(getattr(arm_cfg, 'normalize_action', False)):
        lo = np.asarray(arm_cfg.lower, np.float32)
        hi = np.asarray(arm_cfg.upper, np.float32)
        if lo.ndim == 0 or hi.ndim == 0:
            raise RuntimeError('Arm normalization limits need inspection')
        arm_target = 2 * (arm_target - lo) / (hi - lo) - 1
        arm_target = np.clip(arm_target, -1, 1)
    finger = float(np.clip(np.mean(q[7:9]), -.01, .04))
    if not bool(getattr(grip.config, 'normalize_action', False)):
        grip_target = finger
    else:
        low = np.asarray(grip.config.lower).reshape(-1)[0]
        high = np.asarray(grip.config.upper).reshape(-1)[0]
        grip_target = float(np.clip(2 * (finger-low) / (high-low)-1, -1, 1))
    action = np.concatenate([arm_target, [grip_target]]).astype(np.float32)
    if action.shape != env.action_space.shape[-1:]:
        raise RuntimeError(f'Computed action {action.shape}; env space {env.action_space.shape}')
    return action


def rgb_from_obs(obs):
    try:
        a = obs['sensor_data']['base_camera']['rgb']
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        a = np.asarray(a)
        if a.ndim == 4:
            a = a[0]
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return a
    except (KeyError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['pick','push'], default='push')
    ap.add_argument('--trajectory', type=int, default=6, help='Original trajectory id (traj_N)')
    ap.add_argument('--K', type=int, default=35, help='Number of original observations, handoff at state K-1')
    ap.add_argument('--mode', choices=['oracle','esn'], default='oracle')
    ap.add_argument('--neurons', type=int, default=700)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--split', choices=['validation','train','test'], default='validation')
    ap.add_argument('--video', action='store_true', help='Save GIF of rendered RGB observations')
    ap.add_argument('--max-episode-steps', type=int, default=None,
                    help='Gymnasium TimeLimit for the continuation (default: at least 200).')
    ap.add_argument('--ignore-termination', action='store_true',
                    help='Diagnostic only: continue after termination, but NEVER after truncation.')
    args = ap.parse_args()
    task_id = 'PushCube-v1' if args.task == 'push' else 'PickCube-v1'
    demo = Path.home() / '.maniskill' / 'demos' / task_id / 'motionplanning' / 'trajectory.h5'
    meta_path = demo.with_suffix('.json')
    with open(meta_path) as f:
        meta = json.load(f)
    episode = next((e for e in meta['episodes'] if int(e['episode_id']) == args.trajectory), None)
    if episode is None:
        raise ValueError(f'No episode_id={args.trajectory} in {meta_path}')
    control_mode = episode.get('control_mode', meta['env_info'].get('env_kwargs', {}).get('control_mode'))
    if control_mode != 'pd_joint_pos':
        raise RuntimeError(f'Original control mode is {control_mode!r}, not pd_joint_pos. '
                           'Must convert demonstrations to matching control mode before this experiment.')
    with h5py.File(demo, 'r') as f:
        group = f[f'traj_{args.trajectory}']
        original_actions = np.asarray(group['actions'], np.float32)
        if args.K < 1 or args.K >= len(original_actions) + 1:
            raise ValueError(f'K must be between 1 and {len(original_actions)}')
        # K observed frames mean last observed state is index K-1.
        state_index = args.K - 1
        state_arrays = {kind: {key: np.asarray(d[state_index]).copy()
                               for key, d in group['env_states'][kind].items()}
                        for kind in group['env_states'].keys()}
        reference_final = np.asarray(group['env_states']['actors']['cube'][-1, :3], np.float64)
        ref_goal_actor = 'goal_region' if args.task == 'push' else 'goal_site'
        goal_xyz = np.asarray(group['env_states']['actors'][ref_goal_actor][state_index, :3], np.float64)
    # Get ESN predictions before instantiating simulator, if requested.
    if args.mode == 'esn':
        import parameters as p
        from data_utils import load_split
        from esn import load_model, predict_trajectory
        rows = load_split(args.split, data_path=p.DATA_PATH)
        name = f'{args.task}_{args.trajectory:03d}'
        matches = [r for r in rows if r['id'] == name]
        if not matches:
            raise ValueError(f'{name} not in {args.split}. Use matching split or trajectory id.')
        row = matches[0]
        if int(row['length']) != len(original_actions)+1:
            raise ValueError('Length mismatch between original trajectory and dataset; check replay mapping')
        model_path = p.MODEL_DIR / f'N_{args.neurons}' / f'run_{args.run:02d}' / 'model.npz'
        model = load_model(model_path)
        _, predicted_joints, probability = predict_trajectory(model, row, args.K)
        if predicted_joints is None or not len(predicted_joints):
            raise RuntimeError('This ESN model does not predict joints')
        print(f'ESN push score (ridge, not calibrated probability)={probability:.4f}; predicted joints: {predicted_joints.shape}')
    env_kwargs = dict(meta['env_info'].get('env_kwargs', {}))
    env_kwargs.update(obs_mode='rgb', control_mode='pd_joint_pos', sim_backend='physx_cpu', num_envs=1)
    # The usual 50-step TimeLimit starts at env.reset(), not at state_index.
    # We must allow the COMPLETE predicted continuation to execute.
    min_steps = len(original_actions) - state_index + 10
    time_limit = args.max_episode_steps or max(200, min_steps)
    if time_limit < min_steps:
        raise ValueError(f'--max-episode-steps must be >= {min_steps} for this episode')
    env = gym.make(task_id, max_episode_steps=time_limit, **env_kwargs)
    print(f'Simulator TimeLimit={time_limit}; original remaining actions={len(original_actions)-state_index}')
    frames = []
    try:
        obs, info = env.reset(**episode['reset_kwargs'])
        device = env.unwrapped.device
        state = {group: {name: torch.as_tensor(val, device=device).unsqueeze(0)
                         for name, val in subtree.items()}
                 for group, subtree in state_arrays.items()}
        env.unwrapped.set_state_dict(state)
        # Refresh observation if available. Initial render is diagnostic only.
        if args.video:
            try:
                image = env.render()
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                image = np.asarray(image)
                if image.ndim == 4:
                    image = image[0]
                if image.shape[-1] == 3:
                    frames.append(np.uint8(image))
            except Exception as exc:
                print('Initial rendering unavailable:', exc)
        n_future = len(original_actions) - state_index
        if args.mode == 'esn':
            n_future = min(n_future, len(predicted_joints))
        ever_success = False
        last_success = False
        executed = 0
        stop_reason = 'forecast_exhausted'
        for i in range(n_future):
            if args.mode == 'oracle':
                a = original_actions[state_index + i]
            else:
                a = to_action(predicted_joints[i], env)
            obs, reward, terminated, truncated, info = env.step(a)
            executed += 1
            success = info.get('success', False)
            last_success = to_python(success)
            ever_success |= last_success
            if args.video:
                img = rgb_from_obs(obs)
                if img is not None:
                    frames.append(img)
            if to_python(truncated):
                stop_reason = 'truncated_by_environment'
                break
            if to_python(terminated) and not args.ignore_termination:
                stop_reason = 'terminated_by_environment'
                break
        current_state = env.unwrapped.get_state_dict()
        cube = current_state['actors']['cube']
        if isinstance(cube, torch.Tensor):
            cube = cube.detach().cpu().numpy()
        final_xyz = np.asarray(cube).reshape(-1, np.asarray(cube).shape[-1])[0, :3]
        print(f'Task={task_id} trajectory={args.trajectory} K={args.K} mode={args.mode}')
        print(f'Original final cube xyz: {np.round(reference_final,4)}')
        print(f'Goal xyz: {np.round(goal_xyz,4)}')
        print(f'Simulated final cube xyz: {np.round(final_xyz,4)}')
        print(f'Final cube-to-goal distance XYZ [m]: {np.linalg.norm(final_xyz-goal_xyz):.4f}')
        print(f'Final cube-to-goal distance XY  [m]: {np.linalg.norm(final_xyz[:2]-goal_xyz[:2]):.4f}')
        print(f'Ever success={ever_success} | Success at end={last_success} | Executed steps={executed}/{n_future}')
        print(f'Stop reason: {stop_reason}')
        if stop_reason == 'truncated_by_environment':
            print('WARNING: rollout was truncated; some predictions were not executed.')
        elif executed == n_future:
            print('All available forecast actions were executed (unless terminated at final action).')
        if args.video and frames:
            from PIL import Image
            out = Path('sim_rollouts'); out.mkdir(exist_ok=True)
            path = out / f'{args.task}_{args.trajectory:03d}_K{args.K}_{args.mode}.gif'
            imgs = [Image.fromarray(x).resize((384,384)) for x in frames]
            imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=120, loop=0)
            print('Saved GIF:', path)
    finally:
        env.close()


if __name__ == '__main__':
    main()