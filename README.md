\# 🩺 Disease Diagnosis System



A \*\*Machine Learning-based Disease Diagnosis System\*\* that predicts a possible disease based on symptoms selected by the user. The project uses a trained machine learning model and a Flask web application to provide an easy-to-use interface for disease prediction.



\---



\## 📌 Project Overview



The Disease Diagnosis System is a web-based machine learning application designed to assist users in identifying a possible disease based on their symptoms.



The user selects the symptoms they are experiencing through a web interface. These symptoms are processed and passed to a trained machine learning model, which predicts the most likely disease associated with the selected symptoms.



> \*\*Disclaimer:\*\* This project is intended for educational and demonstration purposes only. It is not a substitute for professional medical diagnosis or treatment.



\---



\## 🎯 Objectives



\* Develop a machine learning-based disease prediction system.

\* Allow users to select symptoms through a web interface.

\* Process the selected symptoms as input features.

\* Use a trained ML model to predict a possible disease.

\* Provide the prediction through a simple Flask web application.

\* Demonstrate the practical use of Machine Learning in healthcare-related applications.



\---



\## 🏗️ System Architecture



```text

&#x20;             ┌─────────────────────┐

&#x20;             │       User          │

&#x20;             └──────────┬──────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;             ┌─────────────────────┐

&#x20;             │   Flask Web Page    │

&#x20;             │   Symptom Selection │

&#x20;             └──────────┬──────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;             ┌─────────────────────┐

&#x20;             │  Input Processing   │

&#x20;             │   \& Preprocessing   │

&#x20;             └──────────┬──────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;             ┌─────────────────────┐

&#x20;             │  Machine Learning   │

&#x20;             │       Model         │

&#x20;             └──────────┬──────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;             ┌─────────────────────┐

&#x20;             │ Disease Prediction  │

&#x20;             └──────────┬──────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;             ┌─────────────────────┐

&#x20;             │ Prediction Display  │

&#x20;             └─────────────────────┘

```



\---



\## 🛠️ Technologies Used



\### Programming Language



\* Python



\### Machine Learning



\* Scikit-learn

\* Pandas

\* NumPy

\* Pickle



\### Web Development



\* Flask

\* HTML

\* CSS

\* Jinja2 Templates



\### Development Tools



\* Visual Studio Code

\* Python Virtual Environment

\* Git

\* GitHub



\---



\## 📂 Project Structure



```text

Disease/

│

├── templates/

│   └── index.html

│

├── .gitignore

├── model.py

├── server.py

├── test.py

├── test\_m.py

│

└── README.md

```



\### Important Local Files



The trained model and dataset are kept locally and are excluded from GitHub because of their large size.



```text

dataset.csv

model.pkl

columns.pkl

```



The Python virtual environment is also excluded:



```text

.venv/

```



\---



\## 📊 Dataset



The project uses a symptom-based disease dataset.



The dataset contains disease information associated with different symptoms. These symptoms are used as input features for machine learning.



\### Dataset Processing



The general workflow includes:



1\. Loading the dataset.

2\. Cleaning the data.

3\. Preparing symptom features.

4\. Encoding/transforming the input data.

5\. Training the machine learning model.

6\. Saving the trained model.

7\. Loading the model in the Flask application.

8\. Predicting the disease from user-selected symptoms.



\---



\## 🤖 Machine Learning Model



A trained machine learning model is used to classify the symptoms and predict a corresponding disease.



The trained model is saved locally as:



```text

model.pkl

```



The feature/column information is stored as:



```text

columns.pkl

```



These files are loaded by the Flask application during prediction.



\---



\## 🔄 Prediction Workflow



```text

User selects symptoms

&#x20;       ↓

Flask receives the input

&#x20;       ↓

Input is converted into model features

&#x20;       ↓

Saved ML model is loaded

&#x20;       ↓

Model processes the symptoms

&#x20;       ↓

Disease is predicted

&#x20;       ↓

Prediction is displayed to the user

```



\---



\## 🌐 Flask Application



The main Flask application is implemented in:



```text

server.py

```



The application provides the web interface and handles the prediction request.



The HTML interface is located in:



```text

templates/index.html

```



\---



\## ⚙️ Installation



\### 1. Clone the Repository



```bash

git clone https://github.com/Tejasai846/Disease.git

```



Move into the project directory:



```bash

cd Disease

```



\---



\### 2. Create a Virtual Environment



Windows:



```powershell

python -m venv .venv

```



\---



\### 3. Activate the Virtual Environment



PowerShell:



```powershell

.venv\\Scripts\\Activate.ps1

```



If PowerShell execution policy prevents activation, use:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

```



Then:



```powershell

.venv\\Scripts\\Activate.ps1

```



\---



\### 4. Install Dependencies



Install the required Python libraries:



```powershell

pip install flask pandas numpy scikit-learn

```



If a `requirements.txt` file is added later, use:



```powershell

pip install -r requirements.txt

```



\---



\## 📦 Required Local Model Files



The GitHub repository does not contain the large model and dataset files.



Before running the application, place the required files in the project directory:



```text

Disease/

│

├── dataset.csv

├── model.pkl

├── columns.pkl

├── server.py

├── model.py

└── templates/

&#x20;   └── index.html

```



\---



\## ▶️ Running the Application



Activate the virtual environment:



```powershell

.venv\\Scripts\\Activate.ps1

```



Start the Flask server:



```powershell

python server.py

```



The terminal should display a local address similar to:



```text

http://127.0.0.1:5000/

```



Open the address in your web browser.



\---



\## 🧪 Testing



The project contains testing scripts:



```text

test.py

test\_m.py

```



These can be used to test the machine learning model and prediction functionality.



Run:



```powershell

python test.py

```



or:



```powershell

python test\_m.py

```



\---



\## 📈 Expected Output



The application allows the user to:



1\. Open the disease diagnosis webpage.

2\. Select symptoms.

3\. Submit the symptoms.

4\. Send the symptoms to the Flask backend.

5\. Process the symptoms using the trained ML model.

6\. Display the predicted disease.



Example:



```text

Selected Symptoms:

\- Fever

\- Headache

\- Fatigue



Predicted Disease:

\[Model Prediction]

```



The actual prediction depends on the trained model and input symptoms.



\---



\## ✨ Features



\* 🩺 Symptom-based disease prediction

\* 🤖 Machine learning integration

\* 🌐 Flask-based web application

\* 🖥️ Simple web interface

\* ⚡ Fast local prediction

\* 📊 Data-driven prediction

\* 🧪 Separate testing scripts

\* 🔒 Large model and dataset files excluded from Git



\---



\## 🔐 Git \& Large Files



The following files are intentionally excluded from GitHub through `.gitignore`:



```text

.venv/

dataset.csv

model.pkl

columns.pkl

```



This is because the dataset and trained model are large files.



The `.gitignore` file contains:



```gitignore

.venv/

dataset.csv

model.pkl

columns.pkl

```



\---



\## 🚀 Future Enhancements



Possible future improvements include:



\* Improve model accuracy.

\* Add multiple machine learning algorithms.

\* Compare model performance.

\* Add probability/confidence information where appropriate.

\* Add a more advanced user interface.

\* Add disease information and general educational resources.

\* Add model evaluation metrics.

\* Add authentication and user accounts.

\* Deploy the application to a cloud platform.

\* Use Git LFS or dedicated model storage for large model files.

\* Add a REST API for predictions.

\* Add automated testing.

\* Add Docker support.



\---



\## ⚠️ Disclaimer



This project is developed for \*\*educational and academic purposes\*\*.



The predictions generated by this application should \*\*not be considered medical advice, diagnosis, or treatment recommendations\*\*.



Users should consult a qualified healthcare professional for medical concerns.



\---



\## 👨‍💻 Project Author



\*\*Tejasai\*\*



GitHub:



\[Tejasai846 GitHub](https://github.com/Tejasai846?utm\_source=chatgpt.com)



Project Repository:



\[Disease Diagnosis System Repository](https://github.com/Tejasai846/Disease.git?utm\_source=chatgpt.com)



\---



\## 📄 License



This project is intended for educational purposes.



You may add an appropriate open-source license such as MIT License if you plan to distribute the project publicly.



