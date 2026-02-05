#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ssc.ssc import sparse_subspace_clustering, sparse_subspace_clustering_cpu

#from torch_kmeans import KMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        attention_boost_q=None, # new
        attention_boost_k=None, # new
        attention_boost_v=None, # new
        save_cluster_path = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                attention_boost_q=attention_boost_q, # new
                attention_boost_k=attention_boost_k, # new
                attention_boost_v=attention_boost_v, # new
                save_cluster_path = save_cluster_path, 
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                attention_boost_q=attention_boost_q, # new
                attention_boost_k=attention_boost_k, # new
                attention_boost_v=attention_boost_v, # new
                save_cluster_path = save_cluster_path, 
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        alpha_q = None,
        alpha_k = None, 
        alpha_v = None,
        num_classes_total = None,
        pca_rank = None, 
        cluster_method = None, 
        rho = None,
        eps = None, 
        save_cluster_path = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        selected = None
        
        # new code
        images_index_selected = (inputs == IMAGE_TOKEN_INDEX)
        
        # 1. 找到 -200 在 inputs 中的位置（视觉 token 的起始位置）
        image_token_start_index = images_index_selected.nonzero(as_tuple=True)[1].item()

        # 2. 计算前面的文本 token 数量
        num_text_tokens_before = image_token_start_index

        # 3. 计算后面的文本 token 数量
        num_text_tokens_after = inputs.shape[1] - num_text_tokens_before - 1
        
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            
        # 4. 计算视觉 token 的数量（即总 token 数量减去前后文本 token 数量）
        num_visual_tokens = inputs_embeds.shape[1] - num_text_tokens_before - num_text_tokens_after

        # 5. 提取 inputs_embeds 中从 image_token_start_index 开始的视觉 token
        vit_embeds = inputs_embeds[:, num_text_tokens_before: (num_text_tokens_before + num_visual_tokens), :]
        
        # 5. 创建新的张量 selected，表示每个 token 是否为视觉 token，初始化为全 False
        selected = torch.zeros(inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)

        # 6. 将视觉 token 对应位置设为 True
        selected[num_text_tokens_before:(num_text_tokens_before + num_visual_tokens)] = True
        
        attention_boost_q = None
        attention_boost_k = None
        attention_boost_v = None

        # ssc_3 (add)
        if cluster_method == 'ssc':
            if os.path.exists(os.path.join(save_cluster_path, "c0.pt")):
                c0 = torch.load(os.path.join(save_cluster_path, "c0.pt")).to(vit_embeds.device)
                c1 = torch.load(os.path.join(save_cluster_path, "c1.pt")).to(vit_embeds.device)
            else:
                # 使用 sparse_subspace_clustering 获取 c0, c1
                c0, c1, _ = sparse_subspace_clustering(vit_embeds.reshape(-1, vit_embeds.shape[-1]).T.float().cpu().numpy(),
                                                         r=pca_rank, n_clusters=num_classes_total, rho=rho, eps=eps) # it's set eps=2e-2 before)

                c0 = torch.tensor(c0)
                c1 = torch.tensor(c1)

                if save_cluster_path is not None:
                    torch.save(c0, os.path.join(save_cluster_path, "c0.pt"))
                    torch.save(c1, os.path.join(save_cluster_path, "c1.pt"))
                    
            relation_sum = c1.to(torch.bfloat16).sum(dim=-1) #.to(inputs_embeds.device) #  / (c0.shape[0])).to(inputs_embeds.device)

            # 假设：labels 是 [2048]，内容是 0~n_classes-1 的类别编号
            labels = c0 #torch.tensor(c0) # torch.tensor([...], dtype=torch.long)  # shape: (2048,)

            # 1. 聚类计数
            class_counts = (torch.bincount(labels, minlength=num_classes_total) / labels.shape[0]) #.to(relation_sum.device)

            relation_sum = (relation_sum * class_counts[labels]).to(inputs_embeds.device)
            relation_sum = (relation_sum - relation_sum.min()) / (relation_sum.max() - relation_sum.min() + 1.e-5)
            relation_sum = relation_sum.to(torch.bfloat16)
            
            boost_factors_q = 1.0 + alpha_q * relation_sum
            boost_factors_k = 1.0 + alpha_k * relation_sum
            boost_factors_v = 1.0 + alpha_v * relation_sum
            del class_counts
            del relation_sum

            attention_boost_q = torch.ones_like(selected, dtype=boost_factors_q.dtype, device=inputs_embeds.device)  # 默认所有为 1.0
            attention_boost_q[selected] = boost_factors_q  # 用视觉增强值替换对应位置 
            
            attention_boost_k = torch.ones_like(selected, dtype=boost_factors_k.dtype, device=inputs_embeds.device)  # 默认所有为 1.0
            attention_boost_k[selected] = boost_factors_k  # 用视觉增强值替换对应位置 

            attention_boost_v = torch.ones_like(selected, dtype=boost_factors_v.dtype, device=inputs_embeds.device)  # 默认所有为 1.0
            attention_boost_v[selected] = boost_factors_v  # 用视觉增强值替换对应位置
        
        if cluster_method is not None:
            return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, attention_boost_q=attention_boost_q, attention_boost_k=attention_boost_k, attention_boost_v=attention_boost_v, save_cluster_path=save_cluster_path, **kwargs)
        else:
            return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, save_cluster_path=save_cluster_path, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_boost_q=None, attention_boost_k=None, attention_boost_v=None, save_cluster_path=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # attention_boost_k = kwargs.get("attention_boost_k", None) # new add
        # attention_boost_v = kwargs.get("attention_boost_v", None) # new add
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if attention_boost_q is not None:
            inputs['attention_boost_q'] = attention_boost_q
        if attention_boost_k is not None:
            inputs['attention_boost_k'] = attention_boost_k
        if attention_boost_v is not None:
            inputs['attention_boost_v'] = attention_boost_v
        if save_cluster_path is not None:
            inputs['save_cluster_path'] = save_cluster_path
            
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
