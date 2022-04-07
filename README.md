# 6CCS3PRJ Individual Project by Jan Marczak

This is a README for an Undergraduate Individual Project **'A thorough attempt to enhance fake news detection through unbiased dataset, explainability and BERT-based models.'**


## Folder Structure

The 'k19029774_marczak_jan_code' folder contains the 4 main components of this project, put in separate folders: 'Google_Scraper', 'Datasets', 'Models' and 'Website'. 

Google_Scraper and Datasets contain Jupyter Notebooks and datasets (csv files) used in datasets analysis and formatting part of this project. Models contains the main Jupyter Notebook for implementing and testing the BERT-based models. Website holds the source code for Streamlit web app implementation. Following sections talk about how to execute these particular components. First of all, it is however recommend to set up a Python environment and install the necessary libraries.

## Setting Up Python Environment

In order to run project's components without issues, a necessary Python libraries and dependencies need to be installed beforehand. It is highly recommended to use a virtual environment such as:

- Anaconda (<https://docs.anaconda.com/anaconda/install/>) (RECCOMENDED) or
- Venv (<https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>). 

The links provide instructions on how to install either of those. 


Once this is done to install the dependencies:
1. Open the terminal and navigate to '6CCS3PRJ_Jan_Marczak_Code' folder.
2. Create a virtual environment:
   - **Anaconda**: ```conda create --name env_name python=3.8``` (RECCOMENDED)
   - **Venv**: ```python3 -m venv env_name ```
3. Activate this virtual environment:
   - **Anaconda**: ```conda activate env_name ``` (RECCOMENDED)
   - **Venv**: ```source env_name/bin/activate ```
4. Install the libraries: ```pip install -r requirements.txt ```

After the libraries are installed (it might take a few minutes) the environment is ready to execute and run various project's components. All of the following sections assume that all the necessary libraries has been installed. It is important to specify the python version


## Google Search Verifier
To run *Google_Search_Verifier.ipynb* file it is recommended to use Jupyter Notebook in a following way:

1. Navigate to 'Google_Scraper' folder.
2. Run ```jupyter notebook Google_Scraper.ipynb ```

Once the notebook opens in a browser the user can browse and execute particular blocks of code.


## Dataset Jupyter Notebooks
'Datasets' folder contains 7 subfolders, each associated and named after a particular dataset that has been analysed or constructed. Each of them contains one *.ipynb* file, which is where the Jupyter Notebook code is provided. Additionally most of them contain 'Initial_datasets' and 'Formatted_datasets' folders that contain necessary csv files either loaded or saved in the Jupyter Notebooks. These are kept if the user wishes to inspect them or run the code.

To run any of these dataset analysis/construction files it is recommended to use Jupyter Notebook in a following way:

1. Navigate to desired folder (for example 'LIAR').
2. Run ```jupyter notebook LIAR.ipynb ```

Once the notebook opens in a browser the user can browse and execute particular blocks of code.


## Models Implementation Jupyter Notebook
*Models_Implementation.ipynb* file is placed in the 'Models' folder and is intented to run in Google Colab, due to their freeily available GPUs. **The code might not run, if executed in Jupyter Notebook.**

In order to inspect the file and run its code, it can be either opened from the following link:

<https://colab.research.google.com/drive/1yC5PjqUOqNCfG-U4G5PsX1CGOY8-E3qQ?usp=sharing>

Or executed individually by:
1. Opening Google Colab <https://colab.research.google.com> in a browser of choice.
2. Picking 'Upload' and choosing a *Models_Implementation.ipynb* file from the local disk.

**The user shoud follow the instructions and comments written in the notebook to ensure everything can be run appropriately.**


## Streamlit Website

In order to run the website, the model needs to be downloaded first. This can be done from this link:

https://drive.google.com/file/d/15Y5kLf3T586_iO3vzWLIWcZ4WyNPa6_M/view?usp=sharing

**In order for the website to work put the downloaded 'roberta-base.pt' file inside Website/model directory!**

Next to run the website:
1. Navigate to 'Website' folder.
2. While having an active environment run ```streamlit run src/app.py```
3. A localhost url will appear in terminal which can be copy-pasted into an internet browser

It is important to run this command withing a 'Website' folder not the 'src' folder! The website might take a minute or two to show up on your first run. **For unknown reasons the website performs very slowly on Safari browser, so please use Google Chrome or Firefox**. 
