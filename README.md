# TS-DP: Reinforcement Speculative Decoding for Temporal Adaptive Diffusion Policy Acceleration

This is the **official code repository** for the paper:

> **TS-DP: Reinforcement Speculative Decoding for Temporal Adaptive Diffusion Policy Acceleration**
>
> [[arXiv:2512.15773]](https://arxiv.org/abs/2512.15773)

TS-DP is the first framework that enables speculative decoding for Diffusion
Policy (DP) with temporal adaptivity. It distills a lightweight Transformer-based
drafter to imitate the base model and replace its costly denoising calls, and
employs an RL-based scheduler that adapts speculative parameters to time-varying
task difficulty—delivering real-time inference while maintaining comparable task
performance.

## Status

🚧 **The code is coming soon and will be open-sourced upon publication.**

We are cleaning up and organizing the codebase (training, inference, and
evaluation scripts, along with configuration files and checkpoints). It will be
released here shortly. Please **star** and **watch** this repository to be
notified when the code becomes available.

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{li2025ts,
  title   = {TS-DP: Reinforcement Speculative Decoding For Temporal Adaptive Diffusion Policy Acceleration},
  author  = {Ye Li and Jiahe Feng and Yuan Meng and Kangye Ji and Chen Tang and Jiajun Fan and Xinwan Wen and Jiawei Li and Wenwu Zhu and Shutao Xia and Xin Wang and Zhi Wang},
  journal = {arXiv preprint arXiv:2512.15773},
  year    = {2025}
}
```

## Contact

For questions about the paper or code, please open an issue in this repository.
