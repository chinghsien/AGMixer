# AGMixer: Age Estimation Using Gender Feature and Improved Ordinal Loss

## Paper

Age estimation can be applied practically with different media and devices across diverse fields. Since the aging process exhibits individual variation and men and women age at different rates, most datasets include both age and gender labels. In this thesis, we propose AGMixer, which leverages gender information to reduce age estimation errors.
We use the Facial Representation Learning (FaRL) pretrained Vision Transformer (ViT) model as a feature extractor and employ a mixer layer for effective feature fusion, achieving lower error rates in age estimation. To utilize gender features and ordinal information in age, we annotated gender labels for CACD2000, CLAP2016, and FG-NET and improved the Ordinal Distance Encoded Regularization (ORDER) loss. We compared our method with others on UTKFace, AFAD, AgeDB, CACD,
CLAP2016, and FG-NET datasets, achieving the lowest Mean Absolute Error (MAE)
across all datasets.

[TBA]()

## Datasets

- [IMDB-Clean](https://github.com/yiminglin-ai/imdb-clean)  
- [UTKFace](https://susanqq.github.io/UTKFace/)  
- [AFAD](https://github.com/John-niu-07/tarball)  
- [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/)  
- [CACD](https://bcsiriuschen.github.io/CARC/)  
- [CLAP2016](https://chalearnlap.cvc.uab.cat/dataset/19/description/#)  
- [FG-NET](https://yanweifu.github.io/FG_NET_data/)  

## Dependencies

- Python 3
- Pytorch
- requirements.txt

## Pretrained Model

We use the pretrained model from [FaRL](https://github.com/FacePerceiver/FaRL) as our backbone.

## Annotations

We provide data annotations for CACD, CLAP2016, and FG-NET with gender labels added based on the[Facial Age Estimation Benchmark](https://github.com/paplhjak/Facial-Age-Estimation-Benchmark). The annotations for the remaining datasets can also be obtained from the same source.

## Train

To train AGMixer with split 0, run the script below.

```=shell
python train.py --split 0
```

To train AGMixer using the all splits, please execute the following command.

```=shell
sh all_split.sh
```

## Test

To test AGMixer using checkpoints, please use the following command.

```=shell
python train.py --split 0 --test
```

## Fine-tune

To fine-tune AGMixer on CACD using the model trained on IMDB-Clean checkpoints, please use the following command.
```=shell
python train.py --fin
```

## Citation

TBA

## Acknowledgement

Thanks to [MWR](https://github.com/nhshin-mcl/MWR?tab=readme-ov-file), [Unified Benchmark](https://github.com/paplhjak/Facial-Age-Estimation-Benchmark), [MLP-Mixer-pytorch](https://github.com/paplhjak/Facial-Age-Estimation-Benchmark), [MLP-Mixer](https://github.com/google-research/vision_transformer?tab=readme-ov-file#mlp-mixer) for releasing their source code.