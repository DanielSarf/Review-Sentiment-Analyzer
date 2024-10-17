# **Review Sentiment Analyzer**  

This project analyzes the sentiment of movie reviews using natural language processing (NLP) techniques. It was developed as part of a learning experience for a university course, with the aim of understanding text classification and machine learning concepts.  

On my end, the model achieved **85.62% accuracy**, though results may vary based on the dataset and training parameters used.  

## **Results:**  
![Results Screenshot](./Screenshots/Output%201.png)  

## **Setup:**  

### **1. Clone the Repository:**  
First, clone this repository to your local machine:
```bash
git clone https://github.com/DanielSarf/Review-Sentiment-Analyzer.git
cd Review-Sentiment-Analyzer
```

### **2. Install Dependencies:**  
Use `pip` to install the required dependencies:
```bash
pip install -r requirements.txt
```

### **3. Download the Dataset:**  
The dataset is not included, nor are the checkpoints, to avoid any licensing issues.  
You can download a suitable dataset that might give similar results:

Here is the link for one:  
[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

alternatively, you can run a kaggle command:  
```bash
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p ./Dataset
```

Make sure to **extract and rename the CSV file (if required)**. This program looks for `./Dataset/IMDB Dataset.csv`.

### **4. Train the Model:**  
After preparing the dataset, run the following command to train the model:
```bash
python train.py
```

### **5. Run the Application:**  
Once the model is trained, you can run the application to test the sentiment analysis:
```bash
python run.py
```
