# Code Repository for AIpparel: A Multimodal Foundation Model for Digital Garments _(CVPR 2025 Highlight)_. 
<p align="center">
  <a href='[https://arxiv.org/abs/2405.04533](https://arxiv.org/abs/2412.03937)'>
    <img src='https://img.shields.io/badge/Arxiv-2405.04533-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://georgenakayama.github.io/AIpparel/'>
  <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'>
  </a>
  <a href='https://huggingface.co/georgeNakayama/AIpparel'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'>
  </a>
</p>

![teaser](assets/imgs/teaser.jpg)

## Dataset 
Please download our GarmentCodeData-Multimodal dataset [here](https://huggingface.co/georgeNakayama/AIpparel), which annotates [GarmentCodeData](https://www.research-collection.ethz.ch/handle/20.500.11850/673889) with editing instructions and textual descriptions. Unzip the downloaded zip file and change the dataset [config file](configs/data_wrapper/dataset/qva_garment_token_dataset_garmentcodedata.yaml).

## Citation

If you are using our model or dataset in your project, consider citing our paper.

```
@article{nakayama2024aipparel,
    title={AIpparel: A Large Multimodal Generative Model for Digital Garments}, 
    author={Kiyohiro Nakayama and Jan Ackermann and Timur Levent Kesdogan 
            and Yang Zheng and Maria Korosteleva and Olga Sorkine-Hornung and Leonidas Guibas
            and Guandao Yang and Gordon Wetzstein},
    journal = {Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```
