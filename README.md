# A Graphical Trendline Analysis Pipeline for Stocks using CNN


#### 2/7/2019

There are two parts right now to this system
#### PART I Data Prep

__data_scraper.py__: provides codes to connect with a financial data API (currently blank and using pre-downloaded AAPL stock price dataset.) 

__generate_graph.py__: reads from the downloaded datasets (stored as csvs) and generate small graphs 


#### PART II CNN

__config.py__: model configurations. 

__model_components.py__: pre-define layers, biases and weights. change this file to customize NN layers 

__pre_precessing.py__: re-write image files into 100 x 100 jpgs

__utils.py__: other tools during model compilation  

__prototype_model.py__: our simple prototype model architect. Change this file to change the model structure.

__trainer.py__: training our model

__predict.py__: predict with new files 

