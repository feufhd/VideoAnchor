# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_internlm2 import InternLM2ForCausalLM

import torch.distributed as dist
import json
import os

from ssc.ssc import sparse_subspace_clustering

#from torch_kmeans import KMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = False # True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=False): # True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', 
             alpha_q=None, alpha_k=None, alpha_v=None, num_classes_total=None, num_classes_selected=None, 
             pca_rank=None, cluster_method=None, rho=None, eps=None, layer_wise_scale=None, boost_layer=None, save_cluster_path=None, verbose=False):
        
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            alpha_q=alpha_q, # new code
            alpha_k=alpha_k, # new code
            alpha_v=alpha_v, # new code
            num_classes_total=num_classes_total, # new code
            num_classes_selected=num_classes_selected, # new code
            pca_rank=pca_rank, # new code
            cluster_method=cluster_method, # new code
            rho=rho, # new code
            eps=eps, # new code
            layer_wise_scale=layer_wise_scale, # new code
            boost_layer=boost_layer, # new code
            save_cluster_path=save_cluster_path, 
            **generation_config
        )
        
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        
        # import pdb; pdb.set_trace()
        
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            alpha_q = None,
            alpha_k = None, 
            alpha_v = None,
            num_classes_total = None,
            num_classes_selected = None, 
            pca_rank = None, 
            cluster_method = None, 
            rho = None,
            eps = None, 
            layer_wise_scale = None,
            boost_layer = None,
            save_cluster_path = None, 
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            
            # import pdb; pdb.set_trace()
            
            attention_boost_q = None
            attention_boost_k = None
            attention_boost_v = None
            
            if False:
                import numpy as np
                np.save(os.path.join(save_cluster_path, 'selected.npy'), selected.cpu().numpy())
                
            # ssc_3 (add)
            if cluster_method == 'ssc_4':
                
                if save_cluster_path is not None:
                    if os.path.exists(os.path.join(save_cluster_path, "c0.pt")):
                        c0 = torch.load(os.path.join(save_cluster_path, "c0.pt")).to(vit_embeds.device)
                        c1 = torch.load(os.path.join(save_cluster_path, "c1.pt")).to(vit_embeds.device)
                        c2 = torch.load(os.path.join(save_cluster_path, "c2.pt")).to(vit_embeds.device)
                    else:
                        # 使用 sparse_subspace_clustering 获取 c0, c1, c2
                        c0, c1, c2 = sparse_subspace_clustering(vit_embeds.reshape(-1, vit_embeds.shape[-1]).T.float().cpu().numpy(),
                                                                 r=pca_rank, n_clusters=num_classes_total, rho=rho, eps=eps) # it's set eps=2e-2 before)
                        
                        if False:
                            umap_vis_path = os.path.join(save_cluster_path, 'umap_vis')
                            os.makedirs(umap_vis_path, exist_ok=True)
                            import numpy as np
                            np.save(os.path.join(umap_vis_path, 'visual_embeds.npy'), vit_embeds.reshape(-1, vit_embeds.shape[-1]).float().cpu().numpy())
                            np.save(os.path.join(umap_vis_path, 'c0.npy'), c0)
                            np.save(os.path.join(umap_vis_path, 'c1.npy'), c1)
                            np.save(os.path.join(umap_vis_path, 'c2.npy'), c2)
                            kmeans_model = KMeans(n_clusters=num_classes_total)
                            if pca_rank > 0:
                                pca = PCA(n_components = pca_rank)
                                labels_kmeans = kmeans_model.fit_predict(pca.fit_transform(vit_embeds.reshape(-1, vit_embeds.shape[-1]).float().cpu().numpy()))
                            else:
                                labels_kmeans = kmeans_model.fit_predict(vit_embeds.reshape(-1, vit_embeds.shape[-1]).float().cpu().numpy())
                            np.save(os.path.join(umap_vis_path, 'labels_kmeans.npy'), labels_kmeans)
                            
                        c0 = torch.tensor(c0)
                        c1 = torch.tensor(c1)
                        c2 = torch.tensor(c2)

                        if save_cluster_path is not None:
                            torch.save(c0, os.path.join(save_cluster_path, "c0.pt"))
                            torch.save(c1, os.path.join(save_cluster_path, "c1.pt"))
                            torch.save(c2, os.path.join(save_cluster_path, "c2.pt"))
                else:
                    # 使用 sparse_subspace_clustering 获取 c0, c1, c2
                    c0, c1, c2 = sparse_subspace_clustering(vit_embeds.reshape(-1, vit_embeds.shape[-1]).T.float().cpu().numpy(),
                                                             r=pca_rank, n_clusters=num_classes_total, rho=rho, eps=eps) # it's set eps=2e-2 before)

                    c0 = torch.tensor(c0)
                    c1 = torch.tensor(c1)
                    c2 = torch.tensor(c2)
                
                relation_sum = c1.to(torch.bfloat16).sum(dim=-1) #.to(inputs_embeds.device) #  / (c0.shape[0])).to(inputs_embeds.device)

                # 假设：labels 是 [2048]，内容是 0~n_classes-1 的类别编号
                labels = c0 #torch.tensor(c0) # torch.tensor([...], dtype=torch.long)  # shape: (2048,)

                # 1. 聚类计数
                class_counts = (torch.bincount(labels, minlength=num_classes_total) / labels.shape[0]) #.to(relation_sum.device)
                
                relation_sum = (relation_sum * class_counts[labels]).to(input_embeds.device)
                relation_sum = (relation_sum - relation_sum.min()) / (relation_sum.max() - relation_sum.min() + 1.e-5)
                relation_sum = relation_sum.to(torch.bfloat16)
                
                
                boost_factors_q = 1.0 + alpha_q * relation_sum
                boost_factors_k = 1.0 + alpha_k * relation_sum
                boost_factors_v = 1.0 + alpha_v * relation_sum
                del class_counts
                del relation_sum
                
                attention_boost_q = torch.ones_like(selected, dtype=boost_factors_q.dtype, device=input_embeds.device)  # 默认所有为 1.0
                attention_boost_q[selected] = boost_factors_q  # 用视觉增强值替换对应位置 
                
                attention_boost_k = torch.ones_like(selected, dtype=boost_factors_k.dtype, device=input_embeds.device)  # 默认所有为 1.0
                attention_boost_k[selected] = boost_factors_k  # 用视觉增强值替换对应位置 
                
                attention_boost_v = torch.ones_like(selected, dtype=boost_factors_v.dtype, device=input_embeds.device)  # 默认所有为 1.0
                attention_boost_v[selected] = boost_factors_v  # 用视觉增强值替换对应位置
            
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            attention_boost_q=attention_boost_q, 
            attention_boost_k=attention_boost_k, # new (2048)
            attention_boost_v=attention_boost_v, # new (2048)
            save_cluster_path=save_cluster_path, 
            selected=selected, 
            # output_attentions=True, # new
            **generate_kwargs,
        )
        
        return outputs

#     @property
#     def lm_head(self):
#         return self.language_model.get_output_embeddings()

#     def get_input_embeddings(self):
#         return self.language_model.get_input_embeddings()

#     def get_output_embeddings(self):
#         return self.language_model.get_output_embeddings()