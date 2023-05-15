# Disaster Response Pipeline Project

This project is part of the Data Engineering section of the Data Science Nanodegree program by Udacity. The goal of this project is to build a pipeline that processes and analyzes disaster response messages. The pipeline includes an ETL (Extract, Transform, Load) process to clean and store the data in a database, as well as an ML (Machine Learning) process to train a classifier for categorizing the messages. The resulting model is then deployed in a web application for users to interact with.

## Project Structure

The project directory contains the following main files:

1. `data`: This directory contains the data processing scripts and the raw data files:

   - `process_data.py`: ETL script to clean and store the data in a SQLite database.
   - `disaster_messages.csv`: CSV file containing the disaster response messages.
   - `disaster_categories.csv`: CSV file containing the categories associated with each message.

2. `models`: This directory contains the machine learning script and the saved model file:

   - `train_classifier.py`: ML script to train a classifier using the cleaned data and save the model as a pickle file.
   - `classifier.pkl`: Saved model file containing the trained classifier.

3. `app`: This directory contains the web application files for visualizing the results:
   - `run.py`: Python script to run the web application.
   - `templates`: Directory containing HTML templates for the web pages.

## Instructions

1. Run the following commands in the project's root directory to set up your database and model:

   - To run the ETL pipeline that cleans the data and stores it in a database:
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

   - To run the ML pipeline that trains the classifier and saves it:
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to the `app` directory:

   ```
   cd app
   ```

3. Run the web app:
   ```
   python run.py
   ```

For more details on the project, please refer to the project's instructions and the source code files.

## Contact Information

Afif Akbar Iskandar
Email: afif_a_iskandar@telkomsel.co.id
