# DBPedia Article Classification: A Full-Stack PoC

This repository contains the source code and documentation for a Proof-of-Concept (PoC) project focused on multi-class text classification of DBPedia articles. The project demonstrates an end-to-end Machine Learning workflow, from data exploration and model experimentation to the deployment of a robust, lightweight model as an interactive web application on Microsoft Azure, complete with CI/CD automation.

## 📋 Project Overview

The primary goal of this project was to evaluate a recent NLP model against established baselines and to deploy the most suitable candidate as a live service.

### Key Objectives

1.  **Model Evaluation:** Compare the performance of a traditional **TF-IDF + Logistic Regression** model against modern Transformer architectures like **DistilBERT** and **ModernBERT**.
2.  **Deployment Strategy:** Build a decoupled, containerized web application with a FastAPI backend and a Streamlit frontend.
3.  **Adaptation to Constraints:** Address real-world deployment challenges, such as container size limits, by making pragmatic, data-driven decisions.
4.  **Automation (CI/CD):** Implement a complete CI/CD pipeline using **GitHub Actions** to automate the build, testing, and deployment process to **Azure Container Apps**.

### The Strategic Pivot

While initial experiments showed that Transformer models like ModernBERT achieved state-of-the-art performance, their resulting Docker image sizes (>10 GB) were prohibitive for a standard cloud deployment.

A key decision was made to **pivot the deployment strategy** to the lightweight yet highly performant **TF-IDF + Logistic Regression** model. This model, when trained on the full dataset, achieved **97.3% accuracy** and a **94.8% macro F1-score**, proving to be the optimal choice for a production-ready, resource-efficient solution.

## 🛠️ Tech Stack

*   **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
*   **Model Tracking:** MLflow
*   **Model Interpretability:** SHAP
*   **Backend API:** FastAPI, Uvicorn
*   **Frontend Dashboard:** Streamlit
*   **Containerization:** Docker, Docker Compose
*   **Cloud & Deployment:** Microsoft Azure (Container Apps, Container Registry), GitHub Actions (CI/CD)

## 📂 Repository Structure

.
├── .github/
│   └── workflows/
│       └── deploy.yml      # GitHub Actions workflow for CI/CD
├── api/
│   └── main.py             # FastAPI application code
├── dashboard/
│   └── app.py              # Streamlit dashboard code
├── models/
│   └── ...                 # (Tracked with Git LFS) Model artifacts
├── .dockerignore           # Specifies files to ignore during Docker build
├── .gitignore              # Specifies files to ignore for Git
├── docker-compose.yml      # For local multi-container development
├── Dockerfile.api          # Dockerfile for the FastAPI service
├── Dockerfile.dashboard    # Dockerfile for the Streamlit service
├── poc_modernbert_exploration.ipynb # Jupyter notebook with all experiments
└── requirements.txt        # Python dependencies

## ⚙️ How to Run Locally

To run the full multi-container application on your local machine, you will need **Docker** and **Docker Compose** installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/M-Carre/OC-NeoBERT-POC.git
    cd OC-NeoBERT-POC
    ```

2.  **Install Git LFS and pull the model file:**
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Build and run the services with Docker Compose:**
    This single command will build the images (if they don't exist) and start both the API and the dashboard containers.
    ```bash
    docker-compose up
    ```

4.  **Access the applications:**
    *   **Dashboard:** Open your browser and navigate to `http://localhost:8501`
    *   **API Docs:** Open your browser and navigate to `http://localhost:8000/docs`

5.  **To stop the application:**
    Press `Ctrl + C` in the terminal, then run:
    ```bash
    docker-compose down
    ```

## 📜 Project Documentation

For a detailed walkthrough of the methodology, analysis, and conclusions, please refer to the project's Jupyter Notebook (`poc_modernbert_exploration.ipynb`) and the formal methodological notes.

---
