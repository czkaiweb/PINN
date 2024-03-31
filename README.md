# Description
This is the repo for GSoC 2024 DeepLense Physics-guided ML task assessment. \
\
**Task I notebook:** \
Task_I_MultiLabel_Classification.ipynb \
**Task V notebook:** \
Task_V_Baseline.ipynb (For baseline) \
Task_V_physics_guided_ML.ipynb (For physics-guided ML with previous lensiformer architecture)

# Main results:
1. CNN and ViT model are fine-tuned from pre-trained model and reach test accuracy of ~94%.
2. Modified Lensiformer is trained with lower resolution and reaches test accuracy of ~64%. The modified Lensiformer model is signifinantly smaller than ViT model and the performance might be limited by the complexity of model architecture.
3. Some proposals to adopt physics-informed methods: \
A. Using simulated lensed image and real lense representation function K(x,y) as input, the encoder transformer is trained to generate K'(x,y) and followed with MLP to classify the lense. To optimize the training, the loss function can be constructed with contribution from both cross entropy and difference between K(x,y) and K'(x,y). \
B. Using simulated lensed image *I_obs(x,y)* and actual source image *I*_source(x,y) as input, the encoder transformer is trained to generate K'(x,y) and reverted image *I_rev(x,y)* is created. The following MLP can use K'(x,y) for multilabel classification and loss function can be constructed with both cross entropy and difference  between *I_rev(x,y)* and *I_source(x,y)*. \
C. Using simulated lensed image *I_obs(x,y)* as input only, the encoder transformer is trained to generate K'(x,y) and reverted image *I_rev(x,y)* is created. The following MLP can use K'(x,y) for multilabel classification and loss function can be constructed with both cross entropy and the concentration of *I_rev(x,y)*. One potential requirement for this approach to work is that dispersion of reverted image with good model-based lense estimation is significant comparing to the intrinsic dispersion of source image. 

# How to
1. Unzip the dataset in the repository as \
`<repo location>/dataset`, remove the `.DS_Store` from `/dataset` and `/dataset/*`
2. Unzip the weights.tar.gz in the repository as `<repo location>/weights`
3. Install the dependency:
```
torch
torchvision
einops
scikit-learn
matplotlib
SciencePlot
tqdm
```
4. For inference, skip the Model Training part in the notebooks.