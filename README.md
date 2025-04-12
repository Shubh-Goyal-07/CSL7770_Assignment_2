# CSL7770 Assignment 2 - Shubh Goyal (B21CS073)

The repository contains the code and report for the second assignment of the course CSL7770 - Speech Understanding.

**Directory Structure**
```
.
├── report.pdf
├── Q1
│   └── ..
├── Q2
│   └── ..
└── README.md
```

The report for both questions is present in the `report.pdf` file.

**How to use this repository**

First, clone the repository to your local machine and navigate to the repository using the following commands:
```bash
git clone https://github.com/Shubh-Goyal-07/CSL7770_Assignment_2.git
cd CSL7770_Assignment_2
```

## Question 1

Navigate to the `Q1` directory to run the code for Question 1.

1. **Navigate to the Q2 directory**
   ```bash
   cd Q1
   ```
    This directory contains all the code files required to run Question 1.

2. **Prepare the environment**
   ```bash
   conda create -n csl77701 -y
   conda activate csl77701
   pip install -r requirements.txt -y
   ```
   
   The above command will create a new conda environment named `csl77701` and install all the required packages in it. It will take a few minutes to complete.

#### Instructions to replicate the results (Speaker Verification and Finetuning)

1. **Navigate to speaker verification directory**
    ```bash
    cd speaker_verification
    ```
    
    This directory contains all the code files required to load, evaluate and finetune the pretrained UniSpeech SAT Large model.

2. **Evaluating and Finetuning the model**

    Run the `main.py` file for both the evaluation and finetuning of the model.  

    ```bash
    python main.py --train is_train --eval_model model_type
    ```

    The `--train` argument is used to specify whether to finetune the model or not. By default, it is set to `False`. If you want to finetune the model, set it to `True`.
    
    The finetuned model and the classifier will be saved in the `models` directory as `best_model.pth` and `classifier.pth` respectively. The log file generated will be named `finetune.log`. Please use it to check the training progress.

    The epochs are set to 4 in the `main.py` file. You can change it to any number you want. The default batch size is set to 32. You can change it in the `main.py` file as well.

    The `--eval_model` argument is used to specify the type of model to be evaluated. By defualt, it is set to ` `, you need to specify either `pretrained` or `finetuned` to evaluate the respective model. If you set the `--train` argument to `True`, the `--eval_model` argument will be ignored.

    The logs for the evaluation will be saved in the `logs` directory as `evaluation_{model_type}.log`. You can check the logs to see the evaluation progress. The EER and TAR@1%FAR will be printed in the logs. 


## Question 2

Navigate to the `Q2` directory to run the code for Question 2.

1. **Navigate to the Q2 directory**
   ```bash
   cd Q2
   ```
    This directory contains all the code files required to run Question 2.

2. **Prepare the environment**
   ```bash
   conda create -n csl77702 -y
   conda activate csl77702
   conda install --file requirements.txt -y -c conda-forge -c nvidia -c pytorch
   ```
   
   The above command will create a new conda environment named `csl77702` and install all the required packages in it. It will take a few minutes to complete.
The code for Question 2 is divided into two tasks:

#### Instructions to replicate the results

1. **Extract Features and Generate spectrograms**
   
   Run the file `processing.py` to extract MFCC features generate the spectrograms for the language audio files. The MFCCs and the generated spectrograms will be saved in the `spectrograms/{language}` for each language directory.
   
   This will also generate a log file named `processing.log` in the `logs` directory.
   
   ```bash
   python processing.py --lang language_name --save_mfcc should_save_bool --num_spectrograms number_of_spectrograms
   ```

   The language_name can be any (multiple separated by space) of the language directories in the `data` directory. By default, it is set to `all`, which will process all the directories in the `data` directory.
   
   The `--save_mfcc` argument is used to save the MFCC features in the `mfccs` directory. By defult, it is set to `False`. If you want to save the MFCC features, set it to `True`. The MFCC features will be saved in the `mfccs/{language}` for each language directory. 
   
   The `--num_spectrograms` argument is used to specify the number of spectrograms to be generated for each language. By default, it is set to 0. Spectrograms will be saved in the `spectrograms/{language}` for each language directory.

   This will also generate a `features` directory containing the MFCC statistics for each audio file in the `.csv` of the respective language.
   
   A `lang_info_plots` directory will also be generated containing statistical feature graphs of all the 13 MFCCs.

   A `lang_stats.json` file will be generated containing the statistical features of all the languages.


2. **Training classifiers**
   
   To train classifiers on the generated data, run the command given below.

   ```bash
   python train.py --model model_name --epochs num_epochs
   ```
   
   The models can be one of the following: `svm`, `rf`, `knn`, `dt`, `nn`.

   The log file generated will be named `train_{model_name}.log`. Please use it to check the training progress. 

   The trained model will be saved as `models/{model_name}.pth/pkl` file.

    The loss and accuracy curve for the neural network model will be saved in the `models` directory itself.

   **Note:** 
    - The argument `--epochs` is useful only when using `nn` model. It will be ignored for other models.
    - The `nn`, `svm`, `rf`, `knn`, `dt` models will use feature statistic csv's generated.
    - `svm` - Support Vector Machine, `rf` - Random Forest, `knn` - K-Nearest Neighbors, `dt` - Decision Tree, `nn` - Neural Network.
  
