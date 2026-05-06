<div align="center">

# MammothModa

**Unified Multimodal Understanding, Generation, and Editing**

<p align="center">
  🌐 <a href="https://mamoda25.github.io/"><b>Project Page</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
  📑 <a href="https://arxiv.org/abs/2511.18262">Mamoda2 Tech Report</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  📑 <a href="https://arxiv.org/abs/2605.02641">Mamoda2.5 Tech Report</a>
</p>

</div>

## Introduction

Mamoda is a family of unified AR-Diffusion models that seamlessly integrate multimodal understanding and generation within a single architecture. One model handles **text-to-image, text-to-video, image editing, video editing, and multimodal understanding**.

## 🎉 News
- 2026-05-06: 🔥**Mamoda2.5** [technical report](https://arxiv.org/abs/2605.02641) is now online! Achieves SOTA on video editing benchmarks. Open-source model weights are under internal review.
- 2026-02-15: 🔥Released **Mamoda2.5** inference code for **Video Generation** and **Video Editing**! Check out our [Project Page](https://mamoda25.github.io/).
- 2025-12-10: 🔥Mamoda2-Dev build upon Qwen3VL-8B supports Image Editing are now available at [HuggingFace](https://huggingface.co/bytedance-research/MammothModa2-Dev). 
- 2025-10-01: 🔥Mamoda2-Preview models are now available at [HuggingFace](https://huggingface.co/bytedance-research/MammothModa2-Preview). **Note: To use the Preview version, please switch to the `qwen25vl` branch.**

## Highlights

<p align="center">
  <img src="./doc/highlight_moe.png" alt="MoE Architecture" width="750" />
</p>

- **Fine-Grained MoE:** 128 routed experts with Top-8 routing — 25B total parameters, only ~3B active per forward pass (~12%), yielding **12x faster inference** than dense models of comparable capacity.
- **Unified Generation & Editing:** A single model for text-to-image, text-to-video, image editing, and video editing — no separate task-specific models needed.
- **SOTA Video Editing:** #1 on OpenVE-Bench (3.86), #1 on FiVE-Bench (87.41), best overall on Reco-Bench.
- **Top-Tier Video Generation:** 61.64 on VBench 2.0, on par with HunyuanVideo 1.5 and LongCat-Video, with only 110s latency.

<p align="center">
  <img src="./doc/highlight_benchmark.png" alt="Benchmark Results" width="750" />
</p>

## Showcases

### Text-to-Video

<table>
  <tr>
    <td width="50%">
      <div align="center">
        <video src="https://github.com/user-attachments/assets/0b5d403b-0565-4c82-a4d4-bce0631f203b" controls="controls" width="100%">
        </video>
        <br><b>Cinematic Shots</b>
      </div>
    </td>
    <td width="50%">
      <div align="center">
        <video src="https://github.com/user-attachments/assets/486bf448-c612-4b1b-953e-9493855df5b4" controls="controls" width="100%">
        </video>
        <br><b>Animal Interaction</b>
      </div>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <div align="center">
        <video src="https://github.com/user-attachments/assets/a5e91bb6-e595-41ac-89b5-f77cf639115e" controls="controls" width="100%">
        </video>
        <br><b>Motion</b>
      </div>
    </td>
    <td width="50%">
      <div align="center">
        <video src="https://github.com/user-attachments/assets/4f855139-4d1e-4a66-b939-982cfa485648" controls="controls" width="100%">
        </video>
        <br><b>Scenery</b>
      </div>
    </td>
  </tr>
</table>

### Video Editing

<table>
  <tr>
    <td width="50%">
      <div align="center">
        <video src="https://github.com/user-attachments/assets/9c5a7328-fed5-4d7a-884c-6dbe4b0d433d" controls="controls" width="100%">
        </video>
        <br><b>Add Backpack</b>
      </div>
    </td>
    <td width="50%">
      <div align="center">
        <video src="https://github.com/user-attachments/assets/d69621da-082c-4103-b4de-cb60a3b77a2c" controls="controls" width="100%">
        </video>
        <br><b>Transform Hand into Robotic Hand</b>
      </div>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <div align="center">
        <video src="https://github.com/user-attachments/assets/65d6baab-a798-4feb-b4dd-8acc40a6f354" controls="controls" width="100%">
        </video>
        <br><b>Ghibli Style</b>
      </div>
    </td>
    <td width="50%">
      <div align="center">
        <video src="https://github.com/user-attachments/assets/f90817db-8967-4731-b63f-ffa746d24212" controls="controls" width="100%">
        </video>
        <br><b>Remove Right Person</b>
      </div>
    </td>
  </tr>
</table>

## Model Family

| Version | Architecture | Capabilities | Details |
|---------|-------------|--------------|---------|
| **Mamoda2.5** | Qwen3-VL + 25B-A3B MoE DiT (E128A8) | Video Gen, Video Edit, Image Edit | [→ mamoda25/](./mamoda25/) |
| **Mamoda2** | Qwen3VL-8B + 3B experts + 2B DiT | Image Gen, Image Edit, Understanding | [→ mamoda2/](./mamoda2/) |

## Citation

```bibtex
@article{shen2025mammothmoda2,
    title={MammothModa2: A Unified AR-Diffusion Framework for Multimodal Understanding and Generation},
    author={Shen, Tao and Wan, Xin and Chen, Taicai and Zhang, Rui and Pan, Junwen and Lu, Dawei and Lei, Fanding and Lu, Zhilin and Yang, Yunfei and Cheng, Chen and She, Qi and Liu, Chang and Sun, Zhenbang},
    journal={arXiv preprint arXiv:2511.18262},
    year={2025},
    url={https://arxiv.org/abs/2511.18262}
}

@article{mamoda25,
    title={Mamoda2.5: Unified Visual Generation and Editing with Fine-Grained MoE DiT},
    journal={arXiv preprint arXiv:2605.02641},
    year={2025},
    url={https://arxiv.org/abs/2605.02641}
}
```

## 🎯 Join Our Team

**Moderation LLM Team @ ByteDance** — We're hiring! Passionate about multimodal AI, computer vision, and MLLM development?

We develop leading MLLMs for content moderation, building infrastructure including model benchmarking, data pipelines, efficient architectures, and training methodologies.

<details>
<summary><b>Recent Publications (2024–2026)</b></summary>

- Pan, J., Zhang, Q., Zhang, R., Lu, M., Wan, X., Zhang, Y., Liu, C., & She, Q. (2025). TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning. ICLR 26.
- Li, Y., Wang, Y., Zhu, Y., Zhao, Z., Lu, M., She, Q., & Zhang, S. (2025). BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models. ICLR 26.
- Li, Z., Qian, D., Su, K., Diao, Q., Xia, X., Liu, C., ... & Yuan, Z. (2025). Bindweave: Subject-consistent video generation via cross-modal integration. ICLR 26.
- Zhang, Q., Cheng, A., Lu, M., Zhuo, Z., Wang, M., Cao, J., Guo, S., She, Q., & Zhang, S. Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs. ICCV 25.
- Xie, R., Du, C., Song, P., & Liu, C. (2025). Muse-vl: Modeling unified vlm through semantic discrete encoding. ICCV 25.
- Zhang, Q., Liu, M., Li, L., Lu, M., Zhang, Y., Pan, J., She, Q., & Zhang, S. (2025). Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs. NeurIPS 25.
- Lin, L., Shi, D., Han, A., Chen, F., Chen, Q., Li, J., ... & Gao, J. (2025). ACT as human: Multimodal large language model data annotation with critical thinking. NeurIPS 25.
- Yu, S., Jin, C., Wang, H., Chen, Z., Jin, S., Zuo, Z., ... & Sun, Q. (2024). Frame-voyager: Learning to query frames for video large language models. ICLR 25.
- Pan, J., Zhang, R., Wan, X., Zhang, Y., Lu, M., & She, Q. (2025). Timesearch: Hierarchical video search with spotlight and reflection for human-like long video understanding. arXiv Preprint arXiv:2504.01407.
- Liu, Z., Pan, J., She, Q., Gao, Y., & Xia, G. (2025). On the Faithfulness of Visual Thinking: Measurement and Enhancement. arXiv Preprint arXiv:2510.23482.
- Zhang, Y., Fan, C.-K., Huang, T., Lu, M., Yu, S., Pan, J., Cheng, K., She, Q., & Zhang, S. (2025). AutoV: Learning to Retrieve Visual Prompt for Large Vision-Language Models. arXiv Preprint arXiv:2506.16112.
- Yuan Zhang, Ming Lu, Junwen Pan, Tao Huang, Kuan Cheng, Chang Liu, Qi She, Shanghang Zhang(2025). ChainV: Atomic Visual Hints Make Multimodal Reasoning Shorter and Better. arXiv Preprint arXiv:2511.17106.
- Shi, H., Liang, J., Xie, R., Wu, X., Chen, C., & Liu, C. (2025). Aquarius: A Family of Industry-Level Video Generation Models for Marketing Scenarios. arXiv preprint arXiv:2505.10584.
- Shen, T., Wan, X., Chen, T., Zhang, R., Pan, J., Lu, D., Lei, F., Lu, Z., Yang, Y., & Cheng, C. (2025). MammothModa2: A Unified AR-Diffusion Framework for Multimodal Understanding and Generation. arXiv Preprint arXiv:2511.18262.
- Qi She, Junwen Pan, Xin Wan, Rui Zhang, Dawei Lu, Kai Huang. (2024). MammothModa: Multi-Modal Large Language Model. arXiv.

</details>

**Contact:** liuchang.lab@bytedance.com
