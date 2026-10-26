"""Extract aligned Panda/cube semantic masks for EXISTING RoboSIM_ESN trajectories.

First run on a single validation ID. No dataset or model is overwritten.
ManiSkill segmentation is privileged simulator supervision (not RGB-only inference).
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa
from PIL import Image


def arr(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def tensor_state(node, i, device):
    if isinstance(node, h5py.Dataset):
        return torch.as_tensor(np.asarray(node[i]), device=device).unsqueeze(0)
    return {key: tensor_state(value, i, device) for key, value in node.items()}


def resized(a, shape, nearest=False):
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    h, w = shape
    return np.asarray(Image.fromarray(a).resize((w, h), mode))


def seg_ids(env):
    from mani_skill.utils.structs import Actor, Link
    mapping = env.unwrapped.segmentation_id_map
    for num, item in sorted(mapping.items()):
        print(f"  {num}: {type(item).__name__} {getattr(item, 'name', '?')}")
    cube = [int(num) for num, item in mapping.items()
            if isinstance(item, Actor) and getattr(item, 'name', '').lower() == 'cube']
    # Panda articulation links may use 'panda_*', 'panda_hand', 'panda_leftfinger', etc.
    panda = [int(num) for num, item in mapping.items()
             if isinstance(item, Link) and ('panda' in getattr(item,'name','').lower() or
                   'finger' in getattr(item,'name','').lower() or 'hand' in getattr(item,'name','').lower())]
    if not cube or not panda:
        raise RuntimeError('Could not resolve cube and Panda segmentation IDs. Review the printed id map.')
    return cube, panda


def image_for_obs(obs):
    cams = obs['sensor_data']
    if 'base_camera' not in cams:
        raise RuntimeError(f'base_camera missing, cameras={list(cams)}')
    cam = cams['base_camera']
    rgb, seg = arr(cam['rgb']), arr(cam['segmentation'])
    if rgb.ndim == 4: rgb = rgb[0]
    if seg.ndim == 4: seg = seg[0]
    if seg.ndim == 3: seg = seg[..., 0]
    return rgb[..., :3].astype('uint8'), seg.astype('int32')


def frame_to_shape(frame, shape, channels=3):
    h, w = shape
    a = np.asarray(frame)
    if a.ndim == 1: a = a.reshape(h, w, channels)
    if a.dtype != np.uint8: a = (np.clip(a, 0, 1) * 255 + .5).astype('uint8')
    return a


def overlay(rgb, cube, panda):
    base = rgb.astype(np.float32).copy()
    out = base.copy()
    out[panda] = .50*base[panda] + .50*np.array([0, 160, 255],np.float32)
    out[cube] = .30*base[cube] + .70*np.array([255, 70, 60],np.float32)
    return out.clip(0,255).astype('uint8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['pick','push'], required=True)
    p.add_argument('--trajectory', type=int, required=True)
    p.add_argument('--split', choices=['train','validation','test'], default='validation')
    p.add_argument('--output', type=Path, default=Path('datasets/maniskill_multimodal/masks_preview'))
    p.add_argument('--max-rgb-mae', type=float, default=.13,
                   help='Abort if RGB render/dataset mismatch; mask alignment must be checked')
    p.add_argument('--frames', type=int, default=0, help='0 = all frames')
    args = p.parse_args()
    import parameters as config
    from data_utils import load_split
    ident = f'{args.task}_{args.trajectory:03d}'
    matching = [r for r in load_split(args.split, data_path=config.DATA_PATH) if r['id'] == ident]
    if len(matching) != 1: raise ValueError(f'{ident} is not in {args.split}')
    row = matching[0]
    taskid = 'PickCube-v1' if args.task=='pick' else 'PushCube-v1'
    path = Path.home()/'.maniskill'/'demos'/taskid/'motionplanning'/'trajectory.h5'
    with open(path.with_suffix('.json')) as f: metadata = json.load(f)
    ep = next(e for e in metadata['episodes'] if int(e['episode_id'])==args.trajectory)
    kwargs = dict(metadata['env_info'].get('env_kwargs',{}))
    kwargs.update(obs_mode='rgb+segmentation', sim_backend='physx_cpu', num_envs=1)
    env = gym.make(taskid, **kwargs)
    try:
        env.reset(**ep['reset_kwargs'])
        cubes, pandas = seg_ids(env)
        masks_cube=[]; masks_robot=[]; sim_rgb=[]; mae=[]
        with h5py.File(path,'r') as f:
            group = f[f'traj_{args.trajectory}']
            total = min(int(row['length']), len(group['env_states']['actors']['cube']))
            if args.frames: total=min(total,args.frames)
            H,W = config.IMAGE_HEIGHT,config.IMAGE_WIDTH
            for i in range(total):
                env.unwrapped.set_state_dict(tensor_state(group['env_states'],i,env.unwrapped.device))
                # Obtain camera output from the restored state; no physics step.
                obs = env.unwrapped.get_obs()
                rgb,seg = image_for_obs(obs)
                cube = resized(np.isin(seg,cubes).astype('uint8')*255,(H,W),True)>0
                robot = resized(np.isin(seg,pandas).astype('uint8')*255,(H,W),True)>0
                r = resized(rgb,(H,W))
                gt = frame_to_shape(row['image'][i],(H,W),config.IMAGE_CHANNELS)
                mae.append(float(np.abs(r.astype('float32')-gt.astype('float32')).mean()/255.))
                sim_rgb.append(r);masks_cube.append(cube);masks_robot.append(robot)
        score = float(np.mean(mae))
        print(f'{ident} frames={len(mae)}  RGB alignment MAE={score:.4f}  maximum={max(mae):.4f}')
        if score > args.max_rgb_mae:
            raise RuntimeError('RGB from restored camera differs from dataset. Masks NOT saved. '
                               'Check camera/shader/replay and temporal alignment; do not force save.')
        output = args.output / ident
        output.mkdir(parents=True,exist_ok=True)
        rgb = np.stack(sim_rgb)
        mc = np.stack(masks_cube)
        mr = np.stack(masks_robot)
        if not mc.any() or not mr.any():
            raise RuntimeError('Empty mask detected. Review segmentation id mapping.')
        np.savez_compressed(output/'masks.npz',robot=mr.astype('uint8'),cube=mc.astype('uint8'),
                            rgb_alignment_mae=np.asarray(mae,np.float32),
                            trajectory_id=ident,split=args.split)
        for i in sorted(set([0,len(rgb)//2,len(rgb)-1])):
            gt=frame_to_shape(row['image'][i],(H,W),config.IMAGE_CHANNELS)
            panels=np.concatenate([gt,overlay(gt,mc[i],mr[i]),rgb[i]],axis=1)
            Image.fromarray(panels).resize((panels.shape[1]*4,panels.shape[0]*4),Image.Resampling.NEAREST).save(output/f'frame_{i:03d}.png')
        (output/'info.json').write_text(json.dumps(dict(task=taskid,trajectory=ident,split=args.split,
                       frames=len(rgb),rgb_alignment_mae=score,robot_pixels=float(mr.mean()),cube_pixels=float(mc.mean()),
                       privileged_simulator_masks=True),indent=2))
        print('Saved',output,'\nMask shape:',mr.shape,'\nMasks are aligned to the RGB observation if check passed.')
    finally:
        env.close()

if __name__=='__main__':main()