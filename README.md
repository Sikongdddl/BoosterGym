# feature
## subgoal: chase ball
method: DQN/SAC
obs space:
obs[0-1]：球相对机器人（机体系）的 XY 位置 delta_xy_body；
obs[2]：平面距离 dist_xy；
obs[3–4]：指向球的方位角的 cos(bearing), sin(bearing)；
obs[5–6]：机体系下的基座线速度 v_body_xy；
obs[7]：朝向球方向的速度分量 speed_toward。

action space:
actions[0-2]:vx, vy, theta
actions[3]: gait_frequency(but freezed here)

## subgoal: pass ball
method: 
obs space:
0-1  p_ball_rel    球相对机体位置 [x,y]（机体系）
2-3  v_ball_rel    球相对机体速度 [vx,vy]（机体系）
4-5  p_goal_rel    目标相对机体位置 [x,y]（机体系）
6-7  goal_dir_rel  球->目标方向单位向量 [ex,ey]（机体系）
8-9  v_base_body   机体线速度 [vx,vy]（机体系）
10   omega_z       机体角速度 ω_z（机体系）
11-12 heading_err  机体朝向相对“球->目标线”的偏差 [sinΔψ, cosΔψ]
13-14 gait_phase   步态相位 [sinφ, cosφ]
15-16 contacts     左右足接触标志 [cL, cR] ∈ [0,1]

action space:
actions[0-2]:vx, vy, theta
actions[3]: gait_frequency(but freezed here)

# usage
## train
(booster) ubuntu@ubun:~/jrWork/booster_gym$ python scripts/train.py --task=PassBallEnv --checkpoint=-1