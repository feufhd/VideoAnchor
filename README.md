<div align="center">
<h3 style="font-size: 1.1em;"><a href="https://www.arxiv.org/abs/2509.25151">
VideoAnchor: Reinforcing Subspace-Structured Visual Cues for Coherent Visual-Spatial Reasoning
</a></h3>
<h3 style="font-size: 1.1em;">(Accepted by ICLR 2026)</h3>

[ZhaoZhi Wang](https://feufhd.github.io/ZhaozhiWang/)<sup>1,2</sup>, [Tong Zhang](https://sites.google.com/view/tong-zhang)<sup>1</sup>, [Mingyue Guo](https://csguomy.github.io/)<sup>2</sup>, [Yaowei Wang](https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN&oi=ao)<sup>1,2</sup>, [Qixiang Ye](https://people.ucas.ac.cn/~0007279?language=en)<sup>1,2</sup>

<sup>1</sup> University of Chinese Academy of Sciences, <sup>2</sup> Peng Cheng Laboratory

</div>

## Getting Started

Please follow the steps below to set up the environment and integrate VideoAnchor for VSI-Bench evaluation:

1. **Environment Setup**  
   Refer to the [thinking-in-space repository](https://github.com/vision-x-nyu/thinking-in-space) and [SSC-Py-CUDA](https://github.com/XHMY/SSC-Py-CUDA) for environment installation instructions, which we gratefully acknowledge. 

2. **Install SSC**  
   ```bash
   cd SSC-Py-CUDA
   pip install -e .
   cd ..
   ```

3. **Integrate Model Wrappers**  
   Move the following files into the lmms_eval/models directory:
- internvl2/internvl2.py
- llava_video/llava_vid.py
- qwen2.5vl/qwen2_5_vl.py

4. **Update Dependencies**  
   Move other files to the right places in *transformers* or the model folder.

5. **Evaluation**  
   Refer to the [thinking-in-space repository](https://github.com/vision-x-nyu/thinking-in-space) for the evaluation.

***We will release the code of VideoAnchor implementation with [FlashBias](https://github.com/thuml/FlashBias) recently.***

<p style="font-size: 1.2em;">
  <strong>
    Please e-mail <a href="mailto:wangzhaozhi22@mails.ucas.ac.cn">me</a> if you have any questions!
  </strong>
</p>

## Citation
```
@article{wang2025videoanchor,
  title={VideoAnchor: Reinforcing Subspace-Structured Visual Cues for Coherent Visual-Spatial Reasoning},
  author={Wang, Zhaozhi and Zhang, Tong and Guo, Mingyue and Wang, Yaowei and Ye, Qixiang},
  journal={arXiv preprint arXiv:2509.25151},
  year={2025}
}
```