# Simple Phishing Classification Model to spot phishing emials

This project detects phishing emails using machine learning. It started as a Jupyter notebook where we cleaned and explored the dataset, processed the text, and trained a Logistic Regression model. We used Optuna to tune hyperparameters and improve model performance. Once the model was trained, we saved it along with the vectorizer for later use.

The project was then turned into a full end-to-end pipeline. We built a FastAPI app to expose a prediction endpoint, containerized everything with Docker, and set up CI/CD using GitHub Actions. The final app is deployed on Google Cloud Run and includes a basic HTML form UI where users can paste an email and get a phishing prediction with confidence. The whole setup is lightweight, scalable, and easy to maintain.

