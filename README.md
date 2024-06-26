Install requirements: 
     pip install -r requirements.txt

Create a data folder in workspace: 

   \data
      ├───bold
      ├───bold_italic
      ├───italic
      └───unbold

Annotating data: 
  - Change the path variable in data_annotation.py file to absolute path to your data folder in workspace
  - Run data_annotation.py

Training:
  - train_ae.ipynb is a notebook to train the auto_encoder
  - train_df.ipynb is a notebook to train the entire model (providing an existed path to autoencoder model in cell 10)
  - Change config variable in cell 2 of both notebook to absolute path to the config folder in workspace 
  - training will return top 2 model with best accuracy
  - training log will be stored in lightning_logs
  
