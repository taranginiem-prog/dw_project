# Amazon Product Analytics Dashboard

## What is included
- `app.py`: Streamlit dashboard
- `requirements.txt`: required Python packages
- `sample_amazon_demo.csv`: clearly labelled synthetic demo data so the dashboard can run immediately

## Recommended setup on Windows
1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ . During installation, tick **Add Python to PATH**.
2. Extract this ZIP into a folder such as `Desktop\amazon_dashboard_project`.
3. Open that folder in File Explorer. Click the address bar, type `cmd`, and press Enter.
4. Run:
   ```
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. A browser tab should open. If not, copy the local URL printed in Command Prompt, usually http://localhost:8501 .
6. The dashboard starts with synthetic demo data. In the left sidebar, upload your real `amazon.csv` to use your course dataset.

## Dataset
Use the `amazon.csv` from your project/Kaggle dataset. This app expects a `rating` column and works best with:
- `product_name`
- `category`
- `discounted_price`
- `actual_price`
- `discount_percentage`
- `rating`
- `rating_count`

The app attempts to normalize common column-name variations and removes currency/percent symbols from numeric values.

## Important academic notes
- The included sample CSV is synthetic and must not be reported as actual Amazon dataset results.
- Classification groups ratings into Low (<3), Medium (3 to <4), and High (4 to 5).
- K-Means standardizes numeric features before clustering.
- Apriori rules use product attributes as a co-occurrence proxy because ordinary product-review datasets generally do not provide real customer basket transactions. Do not claim these are actual items bought together.
- For real recommendations, user-item interaction data and an appropriate recommendation method are needed.
