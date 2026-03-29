# AI_ML Demonstration README File

## Basic Info:
    POC: Mike Pogash
    Origin Date: 2026-03-29
    Revision History: 
    --------------------------------------------------
    ID  |       Date      |    Description
    --------------------------------------------------
    0   |   29-Mar-2026   |  initial drop of: 
                          |      /scripts/linear_regression_neural_network.py
                          |      
## Project Description:


## Purpose: 
    1. The goal of this project is to showcase techincal capabilities with AIML, 
       python, data visualation, and quantitative analysis.  
    
### Current Capabilities
    1. Neural Network (NN) for linear regression programmed from principles
        - see /scripts/linear_regression_neural_network.py
        - uses scikit-learn LinearRegression as source of truth, a tensorflow 
          as a secondary model for performance verification
        - programs a NN from a linear model using first principles
        - visualizes errors between developed NN, scikit-learn Linear regression, 
          and tensorflow NN
        - visualizes progresss of training for the developed NN

### Goal Capabilites:       
    1. See indivudal scripts for a description of desired improvments.

### Limitations:
    1. As of 03-29-2026, repository is limited. Development of the repository
       is ongoing. 

### Required Libraries & APIs:
   pandas, numpy, matplotlib, os, sklearn, tensore flow, datetime, more_itertools

## Developer Notes


### Things to do: 
    1. Wrap common lines into functions
        - synthetic data generator
    2. Save synthetic data as a CSV to showcase reading it in and using it. 
    
### Common Issues:
    1. VS Code was used to access Ubtuntu WSL terminal to run this script. The command pallete, Python: Select 
       Interpreter, was used to select the appropriate Python environment with the necessary libraries installed.

    2. GitHub did not want to talk to the Ubuntu terminal. Needed to call the line below from the Ubuntu terminal: 
        git config --global --add safe.directory '%(prefix)///wsl.localhost/Ubuntu/home/mike/GitHub/AIML_Demonstration'
