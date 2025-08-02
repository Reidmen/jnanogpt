# jNanoGPT

This repository aims at replicating [modded-nangpt](https://github.com/KellerJordan/modded-nanogpt) training with Jax instead of Pytorch. I would call it a speedrun for j-training. 

The aim is to search for the fastest algorithm to use 8 NVIDIA H100 GPUs to train a language model that attains 3.28 cross-entropy loss on the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) validation set.

The target (3.28 validation loss on FineWeb) follows Andrej Karpathy's [GPT-2 replication in llm.c, which attains that loss after running for 45 minutes](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).
The speedrun code also descends from llm.c's [PyTorch trainer](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py), which itself descends from NanoGPT, hence the name of the repo.
Thanks to the efforts of many contributors, this repo now contains a training algorithm which attains the target performance in:
* 3 minutes on 8xH100 (the llm.c GPT-2 replication needed 45)
* 0.73B tokens (the llm.c GPT-2 replication needed 10B)

This improvement in training speed has been brought about by the following techniques:
* Modernized architecture: Rotary embeddings, QK-Norm, and ReLU²
* The Muon optimizer [[writeup](https://kellerjordan.github.io/posts/muon/)] [[repo](https://github.com/KellerJordan/Muon)]
* Untie head from embedding, use FP8 matmul for head, and softcap logits (the latter following Gemma 2)
* Initialization of projection and classification layers to zero (muP-like)
* Skip connections from embedding to every block as well as between blocks in U-net pattern
* Extra embeddings which are mixed into the values in attention layers (inspired by Zhou et al. 2024)
* FlexAttention with long-short sliding window attention pattern (inspired by Gemma 2) and window size warmup

**TODO**

- [x] `basics` folder with relevant transformers implementations.
- [ ] include basics on data, pipeline and tensor parallelism. 
- [x] JAX version of `ref/train_gpt.py` using FSDP. 
- [ ] Timing it until sunset.