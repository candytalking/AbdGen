#Code for "Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings" https://arxiv.org/abs/2310.17451


The program works for conducting rule learning on the Mario dataset.
PyTorch and Swi-Prolog are required for code running.

## Dataset Preparation:
- Preliminary step: download "mario_icons" folder from "https://github.com/EleMisi/VAEL" project, and put it into the "create_dataset" folder.
- The dataset can be constructed with "create_mario_positive.py, create_mario_negative.py" in the "create_dataset" folder.
- These two files are adapted from "https://github.com/EleMisi/VAEL/blob/main/utils/mario_utils/create_mario_dataset.py"
- Dataset saving paths can be modified in the "main" function calls of the two Python files.

## Conduct Learning:
- Running "main.py" in the root folder. 
- Path configurations need to be set properly based on the parser arguments therein.
- More learning configurations can be set in the "exp_config/rule_learning_config.py" file.

