# AI_ML Demonstration README File

## Basic Info:
    POC: Mike Pogash
    Origin Date: 2026-03-29
    Revision History: 
    --------------------------------------------------------------------------------------------
    ID  |       Date      |    Description
    --------------------------------------------------------------------------------------------
    0   |   29-Mar-2026   |  initial baseline of code repo
    1   |   04-Apr-2026   |  successful implementation of custom NN
                          
## Project Description:


### Purpose: 
    - The goal of this project is to showcase techincal capabilities with AIML, 
      python, data visualization,  quantitative analysis, and critical python 
      libraries for data analysis (pandas, numpy, scikit-learn, matplotlib, tensorflow)  

### Current Capabilities
    - Neural Network (NN) for linear regression programmed from principles
        - see /scripts/linear_regression_neural_network.py
        - uses scikit-learn LinearRegression as source of truth, a tensorflow 
          as a secondary model for performance verification
        - programs a NN for a linear model using first principles
        - visualizes errors between customly developed NN, scikit-learn linear
          regression, and a linear regression with tensorflow NN
        - visualizes progresss of training for the developed NN and tensor flow
          as well as compares residuals and resulting fits

### Goal Capabilites:       
    - See indivudal scripts, ./scripts/ for a description of desired improvments.

### Limitations:
    - Repsository started on 03-29-2026. Development of the repository is ongoing. 
    - 04-10-2026. Despite limited duration of development. Goals are accomplished. 
      A custom neural network is programmed from first principles, compared with 
      a standard linear regresion of scikit-learn, and also compared with tensoflow

### Required Libraries & APIs:
    - pandas, numpy, matplotlib, os, sklearn, TensorFlow, datetime, more_itertools
      sys

## Developer Notes


### Things To Do: 
    - make scripts configuration file driven, currently defining inputs within top block of code 
    - Save synthetic data as a CSV to showcase reading it in and using it. 
    
### Common Issues:
    - VSCode defaulting to enable copilot suggestions
         see ./debug/settings_json_configuration.txt

    - VS Code was used to access Ubtuntu WSL terminal to run this script. The command pallete, Python: Select 
      Interpreter, was used to select the appropriate Python environment with the necessary libraries installed.

    - GitHub did not want to talk to the Ubuntu terminal. Needed to call the line below from the Ubuntu terminal: 
      git config --global --add safe.directory '%(prefix)///wsl.localhost/Ubuntu/home/mike/GitHub/AIML_Demonstration'
