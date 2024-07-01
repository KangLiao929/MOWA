# MOWA: Multiple-in-One Image Warping Model

## Introduction
This is the official implementation for [MOWA](https://arxiv.org/abs/2404.10716) (arXiv 2024).

[Kang Liao](https://kangliao929.github.io/), [Zongsheng Yue](https://zsyoaoa.github.io/), [Zhonghua Wu](https://wu-zhonghua.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

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
>  * We show that through multi-task learning, our framework develops a robust generalized warping strategy that gains improved performance across various tasks and even generalizes to unseen tasks.

Check out more visual results and interactions [here](https://kangliao929.github.io/projects/mowa/).

## 📣News
- MOWA has been included in [AI Art Weekly #80](https://aiartweekly.com/issues/80).

## 📝 Changelog

- [x] 2023.04.16: The paper of the arXiv version is online.
- [x] Release the code and pre-trained model.
- [ ] Release a demo for users to try MOWA online.
- [ ] Release an interactive interface to drag the control points and perform customized warpings.

## Installation
Using the virtual environment (conda) to run the code is recommended.
```
conda create -n mowa python=3.8.13
conda activate mowa
pip install -r requirements.txt
```

## Dataset
We mainly explored six representative image warping tasks in this work. The datasets are derived/constructed from previous works. For the convenience of training and testing in one project, we cleaned and arranged these six types of datasets with unified structures and more visual assistance. Please refer to the category and download links in [Datasets](https://github.com/KangLiao929/MOWA/tree/main/Datasets).

## Pretrained Model
Download the pretrained model [here](https://drive.google.com/file/d/1fxQbD1TLoRnW8lG2a8KMinmD6Jlol8EX/view?usp=drive_link) and put it into the ```.\checkpoint``` folder.

## Testing
### Unified Warping and Evaluation on Public Benchmark
Customize the paths of checkpoint and test set, and run:
```
sh scripts/test.sh
```
The warped images and the intermediate results such as the control points and warping flow can be found in the ```.\results``` folder. The evaluated metrics such as PSNR and SSIM are also shown with the task ID.

### Specific Evaluation on Portrait Correction
In the portrait correction task, the ground truth of warped image and flow is unavailable and thus the image quality metrics cannot be evaluated. Instead, the specific metric (ShapeAcc) regarding this task's purpose, i.e., correcting the face distortion, was presented. To reproduce the warping performance on portrait photos, customize the paths of checkpoint and test set, and run:
```
sh scripts/test_portrait.sh
```
The warped images can also be found in the test path.

## Training
Customize the paths of all warping training datasets in a list, and run:
```
sh scripts/train.sh
```

## Demo
TBD

## Acknowledgment
The current version of **MOWA** is inspired by previous specific image warping works such as [RectanglingPano](https://kaiminghe.github.io/publications/sig13pano.pdf), [DeepRectangling](https://github.com/nie-lang/DeepRectangling), [RecRecNet](https://github.com/KangLiao929/RecRecNet), [PCN](https://github.com/uof1745-cmd/PCN), [Deep_RS-HM](https://github.com/DavidYan2001/Deep_RS-HM), [SSPC](https://github.com/megvii-research/Portraits_Correction).

## Citation

```bibtex
@article{liao2024mowa,
  title={MOWA: Multiple-in-One Image Warping Model},
  author={Liao, Kang and Yue, Zongsheng and Wu, Zhonghua and Loy, Chen Change},
  journal={arXiv preprint arXiv:2404.10716},
  year={2024}
}
```

## Contact
For any questions, feel free to email `kang.liao@ntu.edu.sg`.

## License
This project is licensed under [NTU S-Lab License 1.0](LICENSE).
