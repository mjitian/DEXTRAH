cd ~/DEXTRAH/dextrah_lab/rl_games
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1\
  train.py \
    --headless \
    --task=Dextrah-Kuka-Allegro \
    --seed -1 \
    --distributed \
    --num_envs 4096 \
    agent.params.config.minibatch_size=16384 \
    agent.params.config.central_value_config.minibatch_size=16384 \
    agent.params.config.learning_rate=0.0001 \
    agent.params.config.horizon_length=16 \
    agent.params.config.mini_epochs=4 \
    agent.params.config.multi_gpu=True \
    agent.wandb_activate=False \
    env.success_for_adr=0.4 \
    env.objects_dir=visdex_objects \
    env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
    env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
    env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
    env.max_pose_angle=45.0 \
    env.use_cuda_graph=True
