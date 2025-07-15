# CharaConsist

<div align="center">
Official implementation of ICCV 2025 paper - CharaConsist: Fine-Grained Consistent Character Generation

[[Paper](todo)] &emsp; [[Project Page](https://murray-wang.github.io/CharaConsist/)] &emsp; <br>
</div>

# 

## Key Features
- Without any training, CharaConsist can effectively enhance the consistency of text-to-image generation results.
- While maintaining foreground character consistency, CharaConsist can also optionally preserve background consistency, thereby meeting the needs of different application scenarios.
- Built upon the DiT model (FLUX.1), CharaConsist effectively leverages the advantages of the pre-trained model and achieves superior generation quality compared to previous approaches.
- The implementation of CharaConsist includes training-free mask extraction and point matching strategies, which can serve as effective tools for related tasks such as image editing.

## Qualitative Results
<div align="center">Fig. 1 Consistent Character Generation in a Fixed Background.</div>

<a name="fig1"></a>
![Consistent Character Generation in a Fixed Background.](docs/static/images/fg_bg-all.jpg)

<div align="center">Fig. 2 Consistent Character Generation across Different Backgrounds.</div>

<a name="fig2"></a>
![Consistent Character Generation across Different Backgrounds.](docs/static/images/fg_only-all.jpg)

<div align="center">Fig. 3 Story Generation.</div>

<a name="fig3"></a>
![Story Generation.](docs/static/images/story.jpg)

## How to use
### Dependencies and Installation
Only requires that:
- CUDA support
- PyTorch >= 2.0.0
- diffusers

And this released version was tested under the environment specified in requirements.txt.
```bash
conda create --name characonsist python=3.9
conda activate storydiffusion
pip install -U pip
# Install requirements
pip install -r requirements.txt
```

### Quick Start
We provide two ways to use CharaConsist for generating consistent characters:

#### (1) Notebook for Single Example
We provide three Jupyter notebooks: 
- `gen-bg_fg.ipynb`: generating consistent character in a fixed background, as shown in [Fig.1](#fig1).
- `gen-fg_only.ipynb`: generating consistent character across different backgrounds, as shown in [Fig.2](#fig2).
- `gen-mix.ipynb`: generating the same character in partly fixed and partly varying backgrounds, as shown in [Fig.3](#fig3).

Users can refer to the detailed descriptions in the notebooks to familiarize themselves with the entire framework of the method.


#### (2) Script for Batch Generation
We provide a batch generation script in `inference.py`. Its functionality is essentially the same as the notebooks above, but it is more convenient for multiple samples generation. Its input parameters include:

- `init_mode`: Different model initialization methods depending on available GPU memory and number of GPUs.
    | init_mode   | initialization | GPU memory   | GPU number  |
    |--------|------|------------|------|
    | 0   | single GPU   |   37 GB   | 1 |
    | 1   | single GPU, with model cpu offload   |  26 GB  | 1 |
    | 2   | multiple GPUs, memory distribute evenly    |  <= 20 GB | >=2 |
    | 3   | single GPU, with sequantial cpu offload   | 3 GB | 1 |

- `gpu_ids`: The ids of the GPUs to use. When init_mode is set to 0, 1, or 3, only the first GPU in this id list will be used. When init_mode is set to 2, the memory usage will be evenly distributed across all specified GPUs.
- `prompts_file`: Path to the file containing the input prompts. Two examples are provided in the `examples` folder.
- `model_path`: Path to the pre-trained [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) model weights.
- `out_dir`: The path where the output results will be saved.
- `use_interpolate`: Whether to use adaptive token merge. Enabling it improves consistency but increases CPU memory consumption.
- `share_bg`: Whether to preserve the background unchanged
- `save_mask`: Whether to save the automatically extracted masks during the generation process for visualization

Generating consistent character in a fixed background:
```bash
python inference.py \
--init_mode 0 \
--prompts_file examples/prompts-bg_fg.txt \
--model_path path/to/FLUX.1-dev \
--out_dir results/bg_fg \
--use_interpolate --save_mask --share_bg
```

Generating consistent character across different backgrounds:
```bash
python inference.py \
--init_mode 0 \
--prompts_file examples/prompts-fg_only.txt \
--model_path path/to/FLUX.1-dev \
--out_dir results/fg_only \
--use_interpolate --save_mask
```

## BibTeX
If you find CharaConsist useful for your research and applications, please cite using this BibTeX:

```BibTeX
@inproceedings{CharaConsist,
  title={{CharaConsist}: Fine-Grained Consistent Character Generation},
  author={Wang, Mengyu and Ding, Henghui and Peng, Jianing and Zhao, Yao and Chen, Yunpeng and Wei, Yunchao},
  booktitle={ICCV},
  year={2025}
}
```
