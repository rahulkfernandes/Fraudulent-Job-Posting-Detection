# Fraudulent-Job-Posting-Detection

## Description
This project implements a fraudulent job posting classifier based on the title, description and meta data of the job postings. The provided dataset is cleaned, processed and feature engineered to prepare it for training. A soft voting classifier consisting of three models is trained on the pre-processed train data and evaluated against a hold out test set achieving an F1 score of 0.9064 on the test set.

## Dataset
- title: Title of the job posting.
- location: Location of the job.
- description: Job Description.
- requirements: Job requirements.
- telecommuting (Binary): On-Site or WFH/Hybrid 
- has_company_logo (Binary): Presence of the company logo in the posting.
- has_questions (Binary): Presence of questions to the potential applicants.
- fraudulent (Binary): Dependent variable stating the authenticity of the posting.

## Installation
Clone git repository
```
git clone https://github.com/rahulkfernandes/Fraudulent-Job-Posting-Detection.git
```

Install dependencies
```
pip install -r requirements.txt
```

## Usage 
To Train and Test Model
```
python test.py
```

To Train and Test Model 25 Times to Get Metric Statistics (This is just to check the stability of the model)
```
python test2.py
```