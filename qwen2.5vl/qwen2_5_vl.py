import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

import os

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        modality: str = "image",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 8,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        
        self.model_name = pretrained.split('/')[-1]
        
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.modality = modality
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self._processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        #self.sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
        
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split, alpha_q, \
        alpha_k, alpha_v, num_classes_total, num_classes_selected, pca_rank, cluster_method, rho, eps, layer_wise_scale, boost_layer in [reg.args for reg in requests]:

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if self.modality == "image":
                raise NotImplementedError("Image inference for Qwen2VL is not supported yet.")
            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                # 组 messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": f"{video_path}", "nframes": self.max_num_frames},
                            {"type": "text", "text": f"{contexts}"},
                        ],
                    }
                ]

                # 1) chat 模板（单条会话 -> 单个字符串）
                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                # 2) 解析视觉数据
                image_inputs, video_inputs = process_vision_info(messages)

                # 3) （可选）进一步抽帧，确保 <= self.max_num_frames
                if video_inputs is not None and len(video_inputs) > 0:
                    total_frames = video_inputs[0].shape[0]
                    if total_frames > self.max_num_frames:
                        import numpy as np
                        idx = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                        idx = np.unique(idx)
                        if (total_frames - 1) not in idx:
                            idx = np.append(idx, total_frames - 1)
                            idx = np.unique(idx)
                        video_inputs[0] = video_inputs[0][idx]

                # 4) 打包成输入张量（注意 text 传 list，以便 batch 对齐）
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                # 5) 送设备
                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                # 6) 生成参数
                gen_kwargs = dict(
                    eos_token_id=self._tokenizer.eos_token_id,
                    pad_token_id=self._tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=1024, 
                    use_cache=self.use_cache,
                )

                save_cluster_root_path = None
                save_cluster_root_path = 'SET-ROOT-PATH-HERE' + self.model_name +'_' + str(self.max_num_frames) + 'f_' + str(num_classes_total) + 'classes_' + str(rho) + 'rho_' + str(eps) + 'eps'
                if save_cluster_root_path is not None:
                    os.makedirs(save_cluster_root_path, exist_ok=True)

                if cluster_method is not None:
                    if save_cluster_root_path is not None:
                        save_cluster_path = os.path.join(save_cluster_root_path, f"{doc_id:04d}")
                        os.makedirs(save_cluster_path, exist_ok=True)
                
                # 7) 如果你的模型 forward 已经支持这些自定义参数，就一起传
                maybe_extra = dict(
                    cluster_method=cluster_method,
                    alpha_q=alpha_q, alpha_k=alpha_k, alpha_v=alpha_v,
                    num_classes_total=num_classes_total,
                    pca_rank=pca_rank, rho=rho, eps=eps,
                    save_cluster_path=save_cluster_path,
                )
                
                # 8) 生成
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **gen_kwargs, **maybe_extra)

                # 9) 仅取新生成段
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]

                # 10) 解码
                answers = self._processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                output_text = answers[0]                
            else:
                raise NotImplementedError
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."