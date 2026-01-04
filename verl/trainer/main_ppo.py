# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""


from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.utils.reward_score.qa_em import _extract_paired_tags
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import random

def _select_rm_score_fn(data_source):
    supported = {'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'}

    ds = data_source

    if isinstance(ds, (list, tuple)) and len(ds) > 0:
        ds = ds[0]
    if isinstance(ds, np.ndarray):
        tmp = ds.tolist()
        if isinstance(tmp, list) and tmp:
            ds = tmp[0]
        else:
            ds = tmp
    try:
        ds = str(ds).strip().lower()
    except Exception:
        ds = 'unknown'


    return qa_em.compute_score_em


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    @staticmethod
    def _is_unknown_answer(text: str) -> bool:
        """
        Heuristic check whether the answer text should be treated as 'Unknown'.
        We deliberately support both English and a few common Chinese phrasings.
        """
        if text is None:
            return False
        lower = text.strip().lower()
        if not lower:
            return False
        keywords = [
            "unknown",
            "i don't know",
            "i do not know",
            "not sure",
            "cannot answer",
            "can't answer",
            "no sufficient information",
            "insufficient information",
            "don't have enough information",
            "no enough information",
            "no enough information to answer",
        ]
        return any(kw in lower for kw in keywords)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # lengths for placing the reward at the last valid response token
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Build a string for answer extraction:
            #   - Follow original Search-R1 approach: concatenate valid prompt + valid response
            #   - Use full prompt + response to include prompt examples for count-based extraction
            #   - Extract answers by counting matches (prompt has 2 examples: <answer> and </answer> and <answer> Beijing </answer>)
            # decode (matching original Search-R1 logic)
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score function (currently EM-based)
            data_source = data_item.non_tensor_batch['data_source']


            original_answers_list = data.meta_info.get('original_answers', [])
            
            if i < len(original_answers_list) and original_answers_list[i]:
                # Use the pre-extracted original answer from generation phase
                ans1 = original_answers_list[i].strip()
            else:
                # Fallback: Extract from full sequence (for backward compatibility)
                # Use proper tag pairing: match each </answer> with its nearest preceding <answer>
                # This handles incomplete tags correctly (e.g., <answer> text <answer> text </answer>)
                answer_pairs = _extract_paired_tags(sequences_str, '<answer>', '</answer>')
                
                # Prompt has 2 examples: <answer> and </answer> and <answer> Beijing </answer>
                # If len(matches) <= 2, only prompt examples exist, no model-generated answer
                if len(answer_pairs) <= 2:
                    ans1 = ""
                else:
                    # Extract the last answer (before reflection phase if reflection exists)
                    # Check if there's a reflection prompt marker
                    reflection_marker = "You have already given the following answer to the user's question:"
                    reflection_start_pos = sequences_str.find(reflection_marker)
                    
                    if reflection_start_pos != -1:
                        # Find the last <answer> tag BEFORE the reflection prompt
                        ans1 = ""
                        for start_pos, end_pos, content in reversed(answer_pairs):
                            if end_pos < reflection_start_pos:  # This answer is before reflection
                                ans1 = content.strip()
                                break
                    else:
                        # No reflection prompt, use the last answer
                        ans1 = answer_pairs[-1][2].strip() if answer_pairs else ""

            # Calculate Ans1 correctness separately first
            # Ensure qa_em is available (it's imported at module level but sometimes we need to be careful)
            from verl.utils.reward_score import qa_em
            correct1 = bool(ans1) and qa_em.em_check(ans1, ground_truth['target']) == 1
            score1 = 1.0 if correct1 else 0.0

            # Check if we have search_r1_lengths (indicating reflection was enabled)
            search_r1_lengths = data.meta_info.get('search_r1_lengths', [])
            ans2 = None
            
            if search_r1_lengths:
                # --- Reflection Mode (Two-Stage Reward) ---
                
                # 1. Assign Reward for Ans1 (Original Search-R1) at Pos 1
                if i < len(search_r1_lengths):
                    pos1 = int(search_r1_lengths[i]) - 1
                    if 0 <= pos1 < reward_tensor.shape[1]:
                        reward_tensor[i, pos1] = float(score1)

                # 2. Calculate Reward for Ans2 (Reflection)
                final_answer_pairs = _extract_paired_tags(sequences_str, '<final_answer>', '</final_answer>')
                
                # Reflection prompt has 3 examples
                if len(final_answer_pairs) <= 3:
                    score = 0.0
                else:
                    from verl.utils.reward_score import qa_em
                    ans2 = final_answer_pairs[-1][2].strip()
                    correct2 = qa_em.em_check(ans2, ground_truth['target']) == 1
                    is_unknown2 = self._is_unknown_answer(ans2)

                    if correct2:
                        score = 1.0
                    elif (not correct1) and is_unknown2:
                        score = 0.3
                    else:
                        score = 0
            else:
                # --- Standard Mode (No Reflection) ---
                # Just use score1 as the final score
                score = score1

            # Assign final reward (either score2 or score1 depending on mode)
            reward_tensor[i, valid_response_length - 1] = float(score)
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

                print("========== ASSISTANT VIEW (for reward calculation, count-based extraction) ==========")
                print(sequences_str)
                print(f"--------------------------------")
                print(f"Golden answers: {ground_truth['target']}")
                print(f"Extracted answer (ans1): {ans1}")
                print(f"Extracted answer (ans2): {ans2}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }


    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, format_score=0.05)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=2, format_score=0.05)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
