# Post hoc analysis of U-Net latent representations for quality control in stroke lesion segmentation

Repository for the codes related to the paper Quality control through deep learning model latent space interpretability in stroke lesion segmentation.

Main packages can be installed with

```
pip install -r requirements.txt
```

The first part, concerning the deep learning models is in folder `models_training`. The codes `main_X.py` are designed to train your models, according to the three setups in the paper. Evaluation of the models is performed with `evaluation_checkpoint.py`, two lines -indicated with the comments- must be changed for the output constrained model.

The second part, concerning the latent space analysis is in folder `latent_space_analysis`. The codes `PCA.py` and `PaCMAP.py` are meant to project the images latent space features into 2D representation. The 2D features should be gathered in one CSV file with the projection_name_model_dimX as column name to analyze the features. This is done with `LMM_spearman.py` and `mesures_coherence.py` scripts.
