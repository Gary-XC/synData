**Synthetic Banking Data Generator with GANs**

**📌 Project Overview**

This project began as an initiative to build a synthetic SQL database that realistically models the operations of a small bank.
The original scope included generating customer demographics (age, income, education, employment), banking products (accounts, loans, credit cards), and daily transaction data (spending, deposits, bill payments).

While the focus evolved during development, the final implementation centers on a GAN (Generative Adversarial Network) model trained on real bank loan datasets to generate synthetic but realistic loan and financial records.
The synthetic data is pushed into a PostgreSQL database for further use in analysis or downstream applications.

**⚙️ Features**

GAN-based Data Generation
  - Trains on a real bank loans dataset (CSV format).
  - Generates realistic synthetic records.
  - Produces training metrics (d_real, d_fake, g_loss) for evaluation.
    
Configurable Training
  - Number of epochs
  - Batch size
  - Network width
    
Database Integration
  - Output data can be inserted into PostgreSQL for storage and analysis.

**🚀Getting Started**
1. Clone the Repository
  - git clone <repo-url>
  - cd <repo-name>
  - Create a venv and download libraries from the requirements.txt

2. Prepare Your Dataset
  - Place your bank loan dataset in CSV format inside the project directory.
  - Make sure the dataset is clean and properly formatted.

3. Run the GAN Model
  - There are two entry points depending on your GPU:
    - For NVIDIA GPUs:
      - Gan-Test-NVIDIA.py
    - For AMD GPUs:
      - Gan-Test-AMD.py

4. Adjust Training Parameters

Inside the GAN test files, you can modify:
  - epochs – number of training iterations
  - batch_size – size of data batches per step
  - width – controls the size of the neural network

**📊 Training Output**

During training, the model will log three key values:

  - d_real – Discriminator score for real data
  - d_fake – Discriminator score for synthetic (generated) data
  - g_loss – Generator loss, measures how well the generator fools the discriminator

These metrics help track convergence and the realism of generated data.

**🗄️ Future Work/Plans**

This version of this project is being set aside; instead, we are looking to move the vision of this project to the final variation, which uses a different model and system to ultimately reach the same goals. 

The final rendition of the project will use multiple Machine Learning algorithms to help determine which is the best fit for the data using evaluation metrics and feature selection. 
Along with this, we will be creating a usable frontend dashboard to easily access and view the data. 

Find the project here (the link will be added in the future). 

**🛠️ Tech Stack**

GAN implementation --> Python  

Tools --> SciKitLearn, PyTorch, Numpy, Pandas  

Database –-> PostgreSQL

🎯**Skills**
  - ML Engineering
  - Data Processing / Data Pipelining
  - Working with TGAN's
  - Database Management
    
**Data Sets**
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- https://www.kaggle.com/datasets/nikhil1e9/loan-default