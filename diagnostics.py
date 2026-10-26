"""PickCube / PushCube physical continuation diagnostics (no retraining).

Uses original recorded simulator states as an offline reference for TCP/cube/qpos,
then runs the ESN using ONLY the state at K-1 and predicted joint targets.
Reference future states are NEVER used to control the ESN simulation.

Run from RoboSIM_esn with its current sim_rollout.py, parameters.py and ESN model.
"""
import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # register
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from sim_rollout import to_action, to_python, get_state_at


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def xyz(x):
    a = as_numpy(x)
    return a.reshape(-1, a.shape[-1])[0, :3].astype(np.float64).copy()


def observation(env, step, origin):
    e = env.unwrapped
    if not hasattr(e.agent, 'tcp'):
        raise RuntimeError('Panda agent has no agent.tcp; inspect installed ManiSkill TCP API.')
    q = as_numpy(e.agent.robot.get_qpos()).reshape(-1)
    state = e.get_state_dict()
    cube = xyz(state['actors']['cube'])
    tcp = xyz(e.agent.tcp.pose.p)
    if len(q) < 9:
        raise RuntimeError(f'Expected Panda 9 qpos, found {len(q)}')
    width = float(q[7] + q[8])
    return dict(step=int(step), future_step=int(step-origin),
                tcp_x=tcp[0], tcp_y=tcp[1], tcp_z=tcp[2],
                cube_x=cube[0], cube_y=cube[1], cube_z=cube[2],
                tcp_cube_dist=float(np.linalg.norm(tcp-cube)),
                tcp_cube_xy=float(np.linalg.norm(tcp[:2]-cube[:2])),
                finger_width=width, joint_7=float(q[6]),
                finger_left=float(q[7]), finger_right=float(q[8]))


def make_plots(ref, esn, output):
    r = {int(x['step']): x for x in ref}
    pairs = [(v, r[int(v['step'])]) for v in esn if int(v['step']) in r]
    if not pairs:
        return
    t = np.asarray([v['future_step'] for v, _ in pairs], float)
    for slug, cols, ylabel in (
        ('tcp_xyz', ('tcp_x','tcp_y','tcp_z'), 'Position [m]'),
        ('tcp_cube_distance', ('tcp_cube_dist','tcp_cube_xy'), 'Distance [m]'),
        ('finger_width', ('finger_width',), 'Opening [m]'),
        ('cube_height', ('cube_z',), 'Cube center height [m]'),
    ):
        fig, ax = plt.subplots(figsize=(9,4.5))
        for col in cols:
            ax.plot(t, [v[col] for v,_ in pairs], label=f'ESN {col}')
            ax.plot(t, [v[col] for _,v in pairs], '--', label=f'Reference {col}')
        ax.set(xlabel='Future step after K', ylabel=ylabel, title=slug.replace('_',' ').title())
        ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=8)
        fig.tight_layout(); fig.savefig(output/f'{slug}.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot([v['tcp_x'] for _,v in pairs], [v['tcp_y'] for _,v in pairs], '--', label='Reference TCP')
    ax.plot([v['tcp_x'] for v,_ in pairs], [v['tcp_y'] for v,_ in pairs], label='ESN TCP')
    ax.scatter([v['cube_x'] for v,_ in pairs], [v['cube_y'] for v,_ in pairs],
               s=9, alpha=.4, label='ESN cube')
    ax.set(xlabel='World X [m]', ylabel='World Y [m]', title='TCP vs cube (XY)')
    ax.axis('equal');ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(output/'tcp_xy.png', dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task', choices=['pick','push'], default='pick')
    ap.add_argument('--trajectory', type=int, default=6)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--neurons', type=int, default=1000)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--split', choices=['train','validation','test'], default='validation')
    ap.add_argument('--continue-after-success', action='store_true',
                    help='Diagnostic: continue after success termination, never after truncation')
    args=ap.parse_args()

    import parameters as p
    from data_utils import load_split
    from esn import load_model, predict_trajectory

    name=f'{args.task}_{args.trajectory:03d}'
    rows=load_split(args.split, data_path=p.DATA_PATH)
    matching=[r for r in rows if r['id']==name]
    if len(matching)!=1:
        raise ValueError(f'{name} not in split={args.split}; select an episode actually in that split')
    row=matching[0]
    model_path=p.MODEL_DIR/f'N_{args.neurons}'/f'run_{args.run:02d}'/'model.npz'
    model=load_model(model_path)
    _, predicted_q, push_score=predict_trajectory(model, row, args.K)
    if predicted_q is None or len(predicted_q)==0:
        raise RuntimeError('Requires a multimodal ESN that predicts joint positions.')

    task_id='PickCube-v1' if args.task=='pick' else 'PushCube-v1'
    demo=Path.home()/'.maniskill'/'demos'/task_id/'motionplanning'/'trajectory.h5'
    meta=json.loads(demo.with_suffix('.json').read_text())
    eps=[e for e in meta['episodes'] if int(e['episode_id'])==args.trajectory]
    if not eps:
        raise ValueError(f'Episode {args.trajectory} absent from {demo.with_suffix(".json")}')
    ep=eps[0]
    control=ep.get('control_mode',meta.get('env_info',{}).get('env_kwargs',{}).get('control_mode'))
    if control!='pd_joint_pos':
        raise RuntimeError(f'Demo control_mode={control}; expected pd_joint_pos')
    with h5py.File(demo,'r') as f:
        g=f[f'traj_{args.trajectory}']
        actions=np.asarray(g['actions'],np.float32)
        total_states=len(g['env_states']['actors']['cube'])
        if args.K < 1 or args.K >= total_states:
            raise ValueError(f'K must be from 1 to {total_states-1}')
        if int(row['length'])!=len(actions)+1 or total_states!=len(actions)+1:
            raise ValueError('H5 original, processed dataset and actions have incompatible lengths')
        # Gather recorded reference *states* for post-hoc measurement only.
        # Do not pass any of them to the ESN predictor or physical rollout.
        state_index=args.K-1
        recorded_states=[get_state_at(g,j,'cpu') for j in range(state_index,total_states)]

    env_kwargs=dict(meta.get('env_info',{}).get('env_kwargs',{}))
    env_kwargs.update(obs_mode='rgb',control_mode='pd_joint_pos',sim_backend='physx_cpu',num_envs=1)
    remaining=min(len(predicted_q),len(actions)-state_index)
    env=gym.make(task_id, max_episode_steps=max(200,remaining+15),**env_kwargs)
    out=Path('sim_rollouts')/'diagnostics'/p.experiment_tag()/args.split/f'{name}_K{args.K}_N{args.neurons}_run{args.run:02d}'
    out.mkdir(parents=True,exist_ok=True)
    reference=[]; realized=[]; success_any=False; stop='forecast_exhausted'
    try:
        env.reset(**ep['reset_kwargs'])
        device=env.unwrapped.device
        def on_device(node):
            if isinstance(node,dict):
                return {k:on_device(v) for k,v in node.items()}
            return node.to(device)
        for i,s in enumerate(recorded_states):
            env.unwrapped.set_state_dict(on_device(s))
            reference.append(observation(env,state_index+i,state_index))

        env.reset(**ep['reset_kwargs'])
        env.unwrapped.set_state_dict(on_device(recorded_states[0]))
        realized.append(observation(env,state_index,state_index))
        for i in range(remaining):
            action=to_action(predicted_q[i],env)
            _,_,terminated,truncated,info=env.step(action)
            realized.append(observation(env,state_index+i+1,state_index))
            success_any |= to_python(info.get('success',False))
            if to_python(truncated):
                stop='truncated';break
            if to_python(terminated) and not args.continue_after_success:
                stop='terminated';break
    finally:
        env.close()

    refs={x['step']:x for x in reference}
    aligned=[]
    for v in realized:
        r=refs[v['step']]
        dtcp=np.array([v['tcp_'+c]-r['tcp_'+c] for c in ('x','y','z')])
        d_cube=np.array([v['cube_'+c]-r['cube_'+c] for c in ('x','y','z')])
        aligned.append({**v,'tcp_error_x_m':dtcp[0], 'tcp_error_y_m':dtcp[1],
                        'tcp_error_z_m':dtcp[2], 'tcp_error_xyz_m':float(np.linalg.norm(dtcp)),
                        'cube_error_xyz_m':float(np.linalg.norm(d_cube)),
                        'finger_width_error_m':v['finger_width']-r['finger_width'],
                        'reference_tcp_cube_dist':r['tcp_cube_dist'],
                        'reference_finger_width':r['finger_width'],
                        'reference_cube_z':r['cube_z']})
    with (out/'per_frame.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=aligned[0].keys());writer.writeheader();writer.writerows(aligned)
    make_plots(reference,realized,out)
    future=aligned[1:]
    summary={
        'task':args.task,'trajectory':args.trajectory,'K':args.K,'split':args.split,
        'model':str(model_path),'push_score_not_probability':float(push_score),
        'steps_predicted':int(len(predicted_q)),'steps_executed':len(realized)-1,
        'stop_reason':stop,'ever_success':bool(success_any),
        'tcp_mean_error_m':float(np.mean([x['tcp_error_xyz_m'] for x in future])) if future else None,
        'tcp_max_error_m':float(np.max([x['tcp_error_xyz_m'] for x in future])) if future else None,
        'tcp_mean_abs_error_xyz_m':(np.mean(np.abs([[x['tcp_error_'+c+'_m'] for c in 'xyz'] for x in future]),axis=0).tolist() if future else None),
        'max_cube_height_m':float(max(x['cube_z'] for x in realized)),
        'initial_cube_height_m':float(realized[0]['cube_z']),
        'cube_lift_m':float(max(x['cube_z'] for x in realized)-realized[0]['cube_z']),
        'minimum_tcp_cube_distance_m':float(min(x['tcp_cube_dist'] for x in realized)),
        'final_finger_width_m':float(realized[-1]['finger_width']),
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('Saved:',out)
    print('NOTE: reference TCP was measured from saved reference states (offline only).')
    print('NOTE: TCP-center-to-cube-center distance is diagnostic, not a grasp success criterion.')

if __name__=='__main__':
    main()