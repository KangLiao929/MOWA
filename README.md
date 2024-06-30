# MOWA: Multiple-in-One Image Warping Model

## Introduction
This is the official implementation for [MOWA](https://arxiv.org/abs/2404.10716) (arXiv 2024).

Kang Liao, Zongsheng Yue, Zhonghua Wu, Chen Change Loy

S-Lab, Nanyang Technological University


<div align="center">
  <img src="https://github.com/KangLiao929/MOWA/blob/main/assets/teaser.jpg" height="340">
</div>

> ### Why MOWA?
> MOWA is a practical multiple-in-one image warping framework, particularly in computational photography, where six distinct tasks are considered. Compared to previous works tailored to specific tasks, our method can solve various warping tasks from different camera models or manipulation spaces in a single framework. It also demonstrates an ability to generalize to novel scenarios, as evidenced in both cross-domain and zero-shot evaluations.
>  ### Features
>  * The first practical multiple-in-one image warping framework especially in the field of computational photography.
>  * We propose to mitigate the difficulty of multi-task learning by decoupling the motion estimation in both the region level and pixel level.
>  * A prompt learning module, guided by a lightweight point-based classifier, is designed to facilitate task-aware image warpings.

Check out more visual results and interactions [here](https://kangliao929.github.io/projects/mowa/).

## 📣News
- MOWA has been included in [AI Art Weekly #80](https://aiartweekly.com/issues/80).

## 📝 Changelog

- [x] 2023.04.16: The paper of the arXiv version is online.
- [ ] Release the code and pre-trained model.
- [ ] Release a demo for users to try MOWA online.
- [ ] Release an interactive interface to drag the control points and perform customized warpings.

## Installation
Using the virtual environment (conda) to run the code is recommended.
```
conda create -n mowa python=3.8
conda activate mowa
pip install -r requirements.txt
```

## Dataset
We mainly explored six representative image warping tasks in this work. The datasets are derived/constructed from previous works. For the convenience of training and testing in one project, we cleaned and arranged these six types of datasets with unified structures and more visual assistance. Please refer to the category and download links in [Datasets]().

## Pretrained Model
Download the pretrained model [here](https://drive.google.com/file/d/1fxQbD1TLoRnW8lG2a8KMinmD6Jlol8EX/view?usp=drive_link) and put it into the ```.\checkpoint``` folder.

## Citation

```bibtex
@article{liao2024mowa,
  title={MOWA: Multiple-in-One Image Warping Model},
  author={Liao, Kang and Yue, Zongsheng and Wu, Zhonghua and Loy, Chen Change},
  journal={arXiv preprint arXiv:2404.10716},
  year={2024}
}
```

## License
This project is licensed under [NTU S-Lab License 1.0](LICENSE).
