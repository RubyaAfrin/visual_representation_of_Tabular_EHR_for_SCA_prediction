# Visual Representation of Tabular EHRs Data for Predicting Sudden Cardiac Arrest

This project is focused on developing a universal visualization of tabular EHRs data and predicting Sudden Cardiac Arrest (SCA) using deep CNN models.

<p align="center">
  <img src="https://github.com/RubyaAfrin/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/img/workflow_diagram.png" 
</p>
<p align="center"><i>Fig. 1:The overall workflow diagram of the method</i></p>```

## Data Availability
The database used in this study is available at [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) and the extracted tabular EHRs data can be reproduced by [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). 

## Handling Imbalance Image Dataset
Two image datasets are created for handling imbalanced dataset. The detail process of creating balanced datasets is found [here](https://github.com/afrin110203/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/img/flowchart_of_image_dataset_creation.png). 
## Modules
1. [image_generation_from_tabular_EHR_data.py](https://github.com/afrin110203/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/src/image_generation_from_tabular_EHR_data.py) - generates 2D images from EHRs data of MIMIC-III database.
2. [train_test_split.py](https://github.com/afrin110203/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/src/train_test_split.py) - splits the dataset for training (80%) and testing (20%).
3. [ResNet50_with_attention_feature_squeeze.py](https://github.com/afrin110203/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/src/ResNet50_with_attention_feature_squeeze.py) - designs the modified ResNet50 model with attention and feature squeezing mechnism.
4. [model_training.py](https://github.com/afrin110203/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/src/model_training.py) - trains the model by setting hyperparameters.

## Modified ResNet50 Network Architechture
<p align="center">
  <img src="https://github.com/RubyaAfrin/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/img/ResNet50_model_with_attention_feature_squeeze.png" 
</p>
<p align="center"><i>Fig. 2:The ResNet50 model with attention and feature-squeezing mechanism</i></p>

## Performance Evaluation
<p align="center">
  <img src="https://github.com/RubyaAfrin/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/img/performance_analysis_of_deep_CNN_for_SCA_prediction.png" 
</p>
<p align="center"><i>Table I: Performance Comparison of Deep CNN models for SCA Prediction </i></p>

## Requirements
The code is written in Python 3.9 and  the required packages to run the repo is available in [requirements.txt](https://github.com/RubyaAfrin/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/requirements.txt)

## License
The project is licensed under Apache License 2.0 (a permissive license with conditions only requiring preservation of copyright and license notices).
You may see the License at [LICENSE](https://github.com/RubyaAfrin/visual_representation_of_Tabular_EHR_for_SCA_prediction/blob/main/LICENSE) file.
## Contributing
We greatly appreciate any contribution to this project. Before creating a new issue or pull request, 
please read the contribution guidelines and policies in [CONTRIBUTING](https://github.com/afrin110203/LogAnomaliesDetectionDL/blob/main/CONTRIBUTING.md) file.


