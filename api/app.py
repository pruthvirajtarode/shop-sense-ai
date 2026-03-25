import os
from fastapi import FastAPI
import pickle, numpy as np
import pandas as pd

app = FastAPI()

# Absolute paths for Vercel
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "recommender.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "user_item_matrix.csv")

## 📦 Model & Data Preloading
model = None
products_cache = []
categories_cache = ["General"]

try:
    if os.path.exists(MODEL_PATH):
        model = pickle.load(open(MODEL_PATH, "rb"))
        print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    products_path = os.path.join(BASE_DIR, "data", "processed", "products.csv")
    if os.path.exists(products_path):
        df = pd.read_csv(products_path)
        products_cache = df.to_dict("records")
        if "Category" in df.columns:
            categories_cache = sorted(df["Category"].unique().tolist())
            if "All" not in categories_cache:
                categories_cache = ["All"] + categories_cache
        print(f"✅ Loaded {len(products_cache)} products and {len(categories_cache)} categories")
except Exception as e:
    print(f"❌ Error loading products: {e}")

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "ShopSense AI Recommendation API is running",
        "model_loaded": model is not None,
        "products_count": len(products_cache)
    }

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    if model is None:
        return {"error": "Model not trained."}

    users = model.get("users", [])
    if user_id not in users:
        return {"error": f"User {user_id} not found in database. Try IDs like 10001, 10002..."}

    idx = users.index(user_id)
    sim = model["similarity"][idx]
    top_sim_users = np.argsort(sim)[-n-1:-1][::-1]

    # Load matrix only when needed or keep in memory if small
    uif = pd.read_csv(DATA_PATH)
    mat = uif.iloc[:, 1:].values
    product_ids = list(uif.columns[1:])

    recommended = set()
    for u in top_sim_users:
        user_row = mat[u]
        top_items = np.where(user_row > 0)[0]
        for item_idx in top_items:
            recommended.add(product_ids[item_idx])
            if len(recommended) >= n: break
        if len(recommended) >= n: break

    return {
        "user": user_id,
        "top_n": n,
        "recommendations": list(recommended)[:n]
    }

@app.get("/products")
def get_products():
    return products_cache

@app.get("/categories")
def get_categories():
    return categories_cache

