import pandas as pd
import random
import os
from datetime import datetime, timedelta

def create_superstore_dataset():
    # User IDs starting with U0 so they trigger the recommendation engine mapping
    users = [
        {"user_id": f"U{i:03d}", "user_name": name}
        for i, name in zip(
            range(21, 41),
            ["Ethan Wright", "Sophia Turner", "Mason Hill", "Isabella Scott", "William Green",
             "Ava Adams", "James Baker", "Charlotte Nelson", "Benjamin Carter", "Amelia Mitchell",
             "Lucas Perez", "Harper Roberts", "Alexander Turner", "Evelyn Phillips", "Michael Campbell",
             "Abigail Parker", "Daniel Evans", "Emily Edwards", "Matthew Collins", "Elizabeth Stewart"]
        )
    ]
    
    products = [
        # Furniture
        {"product_id": "S001", "product_name": "Ergonomic Office Chair", "category": "Furniture"},
        {"product_id": "S002", "product_name": "Standing Desk - Motorized", "category": "Furniture"},
        {"product_id": "S003", "product_name": "Bookshelf 5-Tier", "category": "Furniture"},
        # Office Supplies
        {"product_id": "S004", "product_name": "Premium Notebook set of 3", "category": "Office Supplies"},
        {"product_id": "S005", "product_name": "Gel Pens - 12 Pack", "category": "Office Supplies"},
        {"product_id": "S006", "product_name": "Desk Organizer Mesh", "category": "Office Supplies"},
        # Tech / Accessories
        {"product_id": "S007", "product_name": "Wireless Mouse Silent", "category": "Technology"},
        {"product_id": "S008", "product_name": "USB-C Hub 7-in-1", "category": "Technology"},
        {"product_id": "S009", "product_name": "Noise Cancelling Earbuds", "category": "Technology"},
        # Decor
        {"product_id": "S010", "product_name": "Artificial Potted Plant", "category": "Decor"},
        {"product_id": "S011", "product_name": "Desk Lamp LED Dimmable", "category": "Decor"},
        # Appliances
        {"product_id": "S012", "product_name": "Mini Fridge 4L", "category": "Appliances"},
        {"product_id": "S013", "product_name": "Single Serve Coffee Maker", "category": "Appliances"},
    ]
    
    sentiments = ["Positive", "Positive", "Positive", "Positive", "Neutral", "Neutral", "Negative"]
    
    interactions = []
    
    start_date = datetime(2024, 1, 1)
    
    for user in users:
        # Each user interacts with 3-5 products
        n_interactions = random.randint(3, 5)
        user_products = random.sample(products, n_interactions)
        
        for prod in user_products:
            # Generate random date
            days_offset = random.randint(0, 90)
            date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
            
            # Generate rating biased towards positive (4-5)
            rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.05, 0.2, 0.4, 0.3])[0]
            
            # Determine sentiment based on rating
            if rating >= 4:
                sentiment = "Positive"
            elif rating == 3:
                sentiment = "Neutral"
            else:
                sentiment = "Negative"
                
            interactions.append({
                "user_id": user["user_id"],
                "user_name": user["user_name"],
                "product_id": prod["product_id"],
                "product_name": prod["product_name"],
                "category": prod["category"],
                "rating": rating,
                "date": date,
                "sentiment": sentiment
            })
            
    # Sort by date
    interactions.sort(key=lambda x: x["date"])
    
    df = pd.DataFrame(interactions)
    
    os.makedirs("data/synthetic", exist_ok=True)
    df.to_csv("data/synthetic/superstore_interactions.csv", index=False)
    print(f"Generated {len(df)} interactions and saved to data/synthetic/superstore_interactions.csv")

if __name__ == "__main__":
    create_superstore_dataset()
