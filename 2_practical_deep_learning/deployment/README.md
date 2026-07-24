# Alligator or Crocodile?

A minimal Streamlit application that classifies photographs as an alligator or a crocodile using a fastai vision model.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository with the contents of this directory at its root.
2. Keep `model.pkl` tracked with Git LFS. The included `.gitattributes` file configures this automatically.
3. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, select the repository, branch, and `app.py` as the main file.
4. Click **Deploy**.

The application runs on CPU and does not require secrets or additional configuration.
