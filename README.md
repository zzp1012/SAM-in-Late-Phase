# SAM-in-Late-Phase

## Abstract

Code release for paper ["Sharpness-Aware Minimization Efficiently Selects Flatter Minima Late In Training"](https://arxiv.org/abs/2410.10373) (accepted by ICLR 2025 Spotlight). 

> Sharpness-Aware Minimization (SAM) has substantially improved the generalization of neural networks under various settings. Despite the success, its effectiveness remains poorly understood. In this work, we discover an intriguing phenomenon in the training dynamics of SAM, shedding light on understanding its implicit bias towards flatter minima over Stochastic Gradient Descent (SGD). Specifically, we find that *SAM efficiently selects flatter minima late in training*. Remarkably, even a few epochs of SAM applied at the end of training yield nearly the same generalization and solution sharpness as full SAM training. Subsequently, we delve deeper into the underlying mechanism behind this phenomenon. Theoretically, we identify two phases in the learning dynamics after applying SAM late in training: i) SAM first escapes the minimum found by SGD exponentially fast; and ii) then rapidly converges to a flatter minimum within the same valley. Furthermore, we empirically investigate the role of SAM during the early training phase. We conjecture that the optimization method chosen in the late phase is more crucial in shaping the final solution's properties. Based on this viewpoint, we extend our findings from SAM to Adversarial Training. 

## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. If you use anaconda3 or miniconda, you can run following instructions to download the required packages in python.
    ```bash
        conda env create -f environment.yml
    ```

---------------------------------------------------------------------------------
Shanghai Jiao Tong University - Email@[zzp1012@sjtu.edu.cn](zzp1012@sjtu.edu.cn)