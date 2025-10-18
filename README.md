## Streamlit Cloud setup

1. Add this repo to Streamlit Community Cloud.
2. In the app settings, create a secrets section with:

   ```
   [kaggle]
   username = "your_kaggle_username"
   key = "your_kaggle_api_key"
   ```

3. (Optional) define `KAGGLE_DATASET` in the app's environment variables to pick a different dataset slug.
4. Deploy – on startup the app will download the Kaggle CSV bundle or fall back to the bundled sample.

The requirements are pinned in `requirements.txt`, so no manual package installs are needed.
