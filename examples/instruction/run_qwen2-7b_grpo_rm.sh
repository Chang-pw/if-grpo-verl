set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ray stop
sleep 5

MODEL_PATH=/mnt/remote-data/zengjie/ckps/qwen_cold_start/checkpoint-224  # replace it with your local file path
RM_MODEL_PATH=/mnt/remote-data/downloads/models/Qwen/Qwen2.5-7B-Instruct
NODES=8 # equal the number of gpus
# export VLLM_ATTENTION_BACKEND=XFORMERS

ray start --head --num-gpus ${NODES} --ray-debugger-external 


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.name=instruction \
    reward_model.reward_manager=instruction_rb \
    reward_model.enable=True \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=8 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['swanlab'] \
    trainer.log_val_generations=60 \
    trainer.project_name='if-verl' \
    trainer.experiment_name='qwen2_7b_grpo_rm' \
    trainer.n_gpus_per_node=${NODES} \
    trainer.save_freq=100 \
    trainer.test_freq=1 \
    trainer.val_before_train=False \
    trainer.total_epochs=5 $@

