cd ~/DEXTRAH/dextrah_lab/distillation
        # NOTE: in general we should try to use a perfect square number of tiles
        python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
          run_distillation.py \
            --distributed \
            --task=tiangong \
            --num_envs 64 env.distillation=True \
            --enable_cameras env.simulate_stereo=True \
            --teacher /home/dodo/DEXTRAH/dextrah_lab/rl_games/logs/tiangong_0.003/2026-01-04_14-49-08/nn/dextrah_lstm.pth  \
            env.img_aug_type="g" \
            env.aux_coeff=10. \
            env.objects_dir="visdex_objects" \
            env.max_pose_angle=45.0 \
            env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
            env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
            env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
            env.use_cuda_graph=False
