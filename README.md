# NetTOP: A light-weight network of orthogonal-plane features for image recognition

**Abstract:**

* In the current light-weight CNN-based networks, convolutional operators are
principally utilized to extract feature maps for image representation. However,
such conventional operation can lead to lack of informative patterns for the
learning process. It is because the operators have just been allocated to convolute on the spatial side of an input tensor. To deal with this deficiency, we
propose a competent model to efficiently exploit the full-side features of a tensor. The proposed model is based on three novel concepts as follows. i) A novel
grouped-convolutional operator is defined to produce complementary features in
consideration of three plane-based volumes that have been correspondingly partitioned subject to three orthogonal planes (TOP) of a given tensor. ii) An effective
perceptron block is introduced to take into account the TOP-based operator for
orthogonal-plane feature extraction. iii) A light-weight backbone of TOP-based
blocks (named NetTOP) is proposed to take advantage of the full-side informative patterns for image representation. Experimental results for image recognition
on benchmark datasets have proved the prominent performance of the proposals.

<u>**Training and validating NetTOP on dataset Stanford Dogs:**</u>

- For training NetTOP on dataset Stanford Dogs and ImageNet:
```
$ python Train_NetTOP_StanfordDogs.py
$ python Train_NetTOP_ImageNet.py
```
- For validating NetTOP on dataset Stanford Dogs and ImageNet::
```
$ python Train_NetTOP_StanfordDogs.py --evaluate
$ python Train_NetTOP_ImageNet.py --evaluate
```
<u>**Note:**</u>
- Subject to your system, modify these files (*.py) to have the right path to dataset

- For the instance of validation of NetTOP on ImageNet, download its trained model at: [Click here](https://drive.google.com/file/d/106AtFXm9mRM1vf-msBl-XUvvSVU-UZLM/view?usp=drive_link). And then locate the downloaded file as ./checkpoints/ImageNet1k/model_best.pth.tar

**Related citations:**

If you use any materials, please cite the following relevant work(s).

```
@article{NetTOPNguyen24,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {NetTOP: A light-weight network of orthogonal-plane features for image recognition},
  journal      = {Machine Learning},
  note         = {(submitted in 2024)}
}
```
