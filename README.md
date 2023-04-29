# CLIP-PAE: Projection-Augmentation Embedding to Extract Relevant Features for a Disentangled, Interpretable, and Controllable Text-Guided Image Manipulation

<p align="center"><a href="https://www.cst.cam.ac.uk/people/cz363">Chenliang Zhou</a>, <a href="https://www.cl.cam.ac.uk/~fz261/">Fangcheng Zhong</a>, <a href="https://www.cl.cam.ac.uk/~aco41/">Cengiz Oztireli</a></p>

<p align="center">Department of Computer Science and Technology<br>University of Cambridge</p>

<p align="center"><a href="https://chenliang-zhou.github.io/CLIP-PAE/">[Project page]</a>      <a href="https://arxiv.org/abs/2210.03919">[Paper]</a></p>

![teaser](docs/img/show_image_display_in_paper.png)


# Abstract
Recently introduced Contrastive Language-Image Pre-Training ([CLIP](https://openai.com/blog/clip/)) bridges images and text by embedding them into a joint latent space. This opens the door to ample literature that aims to manipulate an input image by providing a textual explanation. However, due to the discrepancy between image and text embeddings in the joint space, using text embeddings as the optimization target often introduces undesired artifacts in the resulting images. Disentanglement, interpretability, and controllability are also hard to guarantee for manipulation. To alleviate these problems, we propose to define corpus subspaces spanned by relevant prompts to capture specific image characteristics. We introduce CLIP *projection-augmentation embedding* (PAE) as an optimization target to improve the performance of text-guided image manipulation. Our method is a simple and general paradigm that can be easily computed and adapted, and smoothly incorporated into *any* CLIP-based image manipulation algorithm. To demonstrate the effectiveness of our method, we conduct several theoretical and empirical studies. As a case study, we utilize the method for text-guided semantic face editing. We quantitatively and qualitatively demonstrate that PAE facilitates a more disentangled, interpretable, and controllable image manipulation with state-of-the-art quality and accuracy.

# Usage
Please first install requirements by running
```
pip install -r requirements.txt
```

The semantic face editing experiment can be run by
```
python run_models.py [OPTIONS]
```

Please refer to
```
python run_models.py --help
```
for help in passing the arguments.

In particular, the two most important arguments are `--method` and `--target` (when `--target=text`, this is the naive approach of using text embedding as the target). 
