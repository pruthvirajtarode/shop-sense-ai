from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pickle, numpy as np
import pandas as pd
import os # Added import for os module

app = FastAPI()

# Path configuration for Vercel
API_DIR = os.path.dirname(os.path.abspath(__file__))
# Root dir for index.html serving fallback
ROOT_DIR = os.path.dirname(API_DIR)

# Mount static files
static_dir = os.path.join(API_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
MODEL_PATH = os.path.join(API_DIR, "models", "recommender.pkl")
DATA_PATH = os.path.join(API_DIR, "data", "processed", "user_item_matrix.csv")

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
    products_path = os.path.join(API_DIR, "data", "processed", "products.csv")
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
def read_root():
    index_path = os.path.join(API_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "api_only", "message": "ShopSense AI API is online. Dashboard (index.html) not found in api folder."}

@app.get("/api/status")
def home():
    return {
        "status": "online",
        "message": "ShopSense AI Recommendation API is running",
        "model_loaded": model is not None,
        "products_count": len(products_cache)
    }

@app.get("/api/recommend/{user_id}")
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

@app.get("/api/products")
def get_products():
    return products_cache

@app.get("/api/categories")
def get_categories():
    return categories_cache

