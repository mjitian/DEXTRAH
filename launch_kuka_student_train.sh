cd ~/DEXTRAH/dextrah_lab/distillation
        # NOTE: in general we should try to use a perfect square number of tiles
        python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
          run_distillation.py \
            --distributed \
            --task=Dextrah-Kuka-Allegro \
            --num_envs 12 env.distillation=True \
            --enable_cameras env.simulate_stereo=False \
            --teacher  /home/dodo/DEXTRAH/dextrah_lab/rl_games/logs/kuka/nn/dextrah_lstm.pth \
            env.img_aug_type="depth" \
            env.aux_coeff=10. \
            env.objects_dir="visdex_objects" \
            env.max_pose_angle=45.0 \
            env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
            env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
            env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
            env.use_cuda_graph=False
