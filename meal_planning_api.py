from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pymongo
import os
import re
import json
import time
from datetime import date
import random
from dotenv import load_dotenv
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# MongoDB Connection
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not found in environment variables.")
client = pymongo.MongoClient(MONGODB_URI)
db = client["Healthy_Aahar"]
users_collection = db["users"]
recipes_collection = db["recipes"]
meal_plans_collection = db["meal_plans"]
nutrition_logs_collection = db["nutrition_logs"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

class UserProfile(BaseModel):
    uid:str
    name: str
    age: int
    gender: str
    weight: float
    height: float
    activity_level: str  # "sedentary", "light", "moderate", "active"
    goal: str  # "weight_loss", "maintenance", "weight_gain"

def calculate_bmr(weight, height, age, gender):
    if gender.lower() == "male":
        return 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        return 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725
    }
    return bmr * activity_multipliers.get(activity_level.lower(), 1.2)

def generate_recipe(calories, goal):
    prompt = f"""
    Create a unique Indian recipe that matches the following criteria:
    - Calories: {calories} kcal
    - Goal: {goal} (e.g., weight loss, maintenance, weight gain)
    Provide the response strictly in JSON format with:
    {{
        "TranslatedRecipeName": "Recipe Name",
        "Cuisine": "Cuisine Type",
        "TotalTimeInMins": 00,
        "TranslatedInstructions": "Step-by-step instructions",
        "Calories": {calories},
        "Protein": according to the recipe ,
        "Fat": according to the recipe ,
        "Carbs": according to the recipe
    }}
    """
    try:
        response = gemini_model.generate_content(prompt)
        recipe_data = json.loads(response.text.strip())
        recipes_collection.insert_one(recipe_data)
        return recipe_data
    except Exception as e:
        print(f"Error generating recipe: {e}")
        return None

@app.post("/register_user")
def register_user(user: UserProfile):
    existing_user = users_collection.find_one({"uid": user.uid})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    bmr = calculate_bmr(user.weight, user.height, user.age, user.gender)
    tdee = calculate_tdee(bmr, user.activity_level)
    
    user_data = user.dict()
    user_data["bmr"] = bmr
    user_data["tdee"] = tdee
    
    users_collection.insert_one(user_data)
    return {"message": "User registered successfully", "bmr": bmr, "tdee": tdee}

@app.get("/get_user/{uid}")
def get_user(uid: str):
    user = users_collection.find_one({"uid": uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["_id"] = str(user["_id"])
    return jsonable_encoder(user)

class IngredientRecipeRequest(BaseModel):
    uid: str
    ingredients: list[str]

@app.post("/generate_recipe_by_ingredients")
def generate_recipe_by_ingredients(request: IngredientRecipeRequest):
    user = users_collection.find_one({"uid": request.uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Pull user-specific data
    goal = user["goal"]
    tdee = user["tdee"]
    calorie_target = tdee / 3  # 3 meals per day logic

    # Adjust based on goal
    if goal == "weight_loss":
        calorie_target -= 100
    elif goal == "weight_gain":
        calorie_target += 100

    # Call Gemini to generate the recipe
    recipe = generate_recipe_with_ingredients(request.ingredients, calorie_target, goal)

    if recipe:
        recipe["_id"] = str(recipe.get("_id", ""))
        return jsonable_encoder(recipe)
    else:
        raise HTTPException(status_code=500, detail="Failed to generate recipe")

def generate_recipe_with_ingredients(ingredients, calories, goal):
    ingredient_list = ", ".join(ingredients)
    prompt = f"""
    Create a unique Indian recipe using the following ingredients: {ingredient_list}
    Match the following criteria:
    - Calories: around {calories} kcal
    - Goal: {goal} (e.g., weight loss, maintenance, weight gain)

    Provide the response strictly in JSON format:
    {{
        "TranslatedRecipeName": "Recipe Name",
        "Cuisine": "Cuisine Type",
        "TotalTimeInMins": 00,
        "TranslatedInstructions": "Step-by-step instructions",
        "Calories": estimated calorie count,
        "Protein": estimated protein in grams,
        "Fat": estimated fat in grams,
        "Carbs": estimated carbs in grams
    }}
    """

    try:
        response = gemini_model.generate_content(prompt)
        
        # Try to extract text safely based on your SDK
        response_text = getattr(response, "text", None)
        if not response_text and hasattr(response, "candidates"):
            response_text = response.candidates[0].content.parts[0].text
        
        if not response_text:
            raise ValueError("Empty or invalid Gemini response")
        
        print("Raw Gemini response:")
        print(response_text)

        recipe_data = json.loads(response_text.strip())
        recipes_collection.insert_one(recipe_data)
        return recipe_data
    except Exception as e:
        print(f"Error generating recipe with ingredients: {e}")
        return None


@app.get("/generate_meal_plan/{user_name}")
def generate_meal_plan(user_name: str):
    user = users_collection.find_one({"name": user_name})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    daily_calories = user["tdee"]
    calorie_target = daily_calories / 3  # Assume 3 meals per day
    goal = user["goal"]

    # Adjust calorie targets for goals
    if goal == "weight_loss":
        calorie_target -= 100
    elif goal == "weight_gain":
        calorie_target += 100

    meal_plan = []
    meals_cursor = recipes_collection.find({"Calories": {"$gte": calorie_target - 50, "$lte": calorie_target + 50}})
    meals_list = list(meals_cursor)

    # Shuffle the meals list to ensure variety on each API call
    random.shuffle(meals_list)
    
    # If not enough meals, generate AI-based recipes
    while len(meals_list) < 3:
        ai_recipe = generate_recipe(calorie_target, goal)
        if ai_recipe:
            meals_list.append(ai_recipe)
        else:
            break
    
    if not meals_list:
        raise HTTPException(status_code=404, detail="No suitable meals found")
    
    # Select different meals for the plan
    for _ in range(3):
        meal = meals_list.pop(0) if meals_list else None
        if meal:
            meal["_id"] = str(meal["_id"])  # Convert ObjectId to string
            meal_plan.append(meal)
    
    meal_plan_data = {"user_name": user_name, "meal_plan": meal_plan}
    meal_plans_collection.insert_one(jsonable_encoder(meal_plan_data))
    
    return jsonable_encoder(meal_plan_data)

class SwapMealRequest(BaseModel):
    meal_index: int

@app.put("/swap_meal/{user_name}")
def swap_meal(user_name: str, request: SwapMealRequest):
    user_meal_plan = meal_plans_collection.find_one({"user_name": user_name})
    if not user_meal_plan:
        raise HTTPException(status_code=404, detail="Meal plan not found")

    meal_plan = user_meal_plan["meal_plan"]
    if request.meal_index >= len(meal_plan):
        raise HTTPException(status_code=400, detail="Invalid meal index")

    old_meal = meal_plan[request.meal_index]
    query = {"Calories": {"$gte": old_meal["Calories"] - 50, "$lte": old_meal["Calories"] + 50}}
    new_meal = recipes_collection.find_one(query)

    if new_meal:
        new_meal["_id"] = str(new_meal["_id"])  # Convert ObjectId to string
        meal_plan[request.meal_index] = new_meal
        meal_plans_collection.update_one({"user_name": user_name}, {"$set": {"meal_plan": meal_plan}})
        return jsonable_encoder({"message": "Meal swapped successfully", "new_meal": new_meal})

    return {"message": "No suitable meal found for swapping"}

class LogMealRequest(BaseModel):
    uid: str
    meal_id: str  # ID of the recipe from your recipe collection
    date: date    # Format: "YYYY-MM-DD"

@app.post("/log_meal")
def log_meal(request: LogMealRequest):
    user = users_collection.find_one({"uid": request.uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        recipe = recipes_collection.find_one({"_id": pymongo.ObjectId(request.meal_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid meal_id format")

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    meal_data = {
        "meal_id": str(recipe["_id"]),
        "meal_name": recipe["TranslatedRecipeName"],
        "calories": recipe["Calories"],
        "protein": recipe["Protein"],
        "fat": recipe["Fat"],
        "carbs": recipe["Carbs"]
    }

    date_str = str(request.date)

    # Check if there's already a log for this user & date
    log = nutrition_logs_collection.find_one({"uid": request.uid, "date": date_str})

    if log:
        log["meals"].append(meal_data)
        log["total"]["calories"] += meal_data["calories"]
        log["total"]["protein"] += meal_data["protein"]
        log["total"]["fat"] += meal_data["fat"]
        log["total"]["carbs"] += meal_data["carbs"]

        nutrition_logs_collection.update_one(
            {"uid": request.uid, "date": date_str},
            {"$set": {
                "meals": log["meals"],
                "total": log["total"]
            }}
        )
    else:
        nutrition_logs_collection.insert_one({
            "uid": request.uid,
            "date": date_str,
            "meals": [meal_data],
            "total": {
                "calories": meal_data["calories"],
                "protein": meal_data["protein"],
                "fat": meal_data["fat"],
                "carbs": meal_data["carbs"]
            }
        })

    return {"message": "Meal logged successfully"}

@app.get("/daily_nutrition/{uid}/{date}")
def get_daily_nutrition(uid: str, date: str):
    log = nutrition_logs_collection.find_one({"uid": uid, "date": date})
    if not log:
        raise HTTPException(status_code=404, detail="No nutrition log found for this date")

    log["_id"] = str(log["_id"])
    return jsonable_encoder(log)

class CustomMealRequest(BaseModel):
    uid: str
    date: date
    food_description: str

@app.post("/log_custom_meal")
def log_custom_meal(request: CustomMealRequest):
    user = users_collection.find_one({"uid": request.uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    prompt = f"""
    Estimate the nutritional values for the following meal: "{request.food_description}"
    Return the result strictly in JSON format like:
    {{
        "meal_name": "Food description",
        "calories": integer,
        "protein": grams as integer,
        "fat": grams as integer,
        "carbs": grams as integer
    }}
    """
    try:
        response = gemini_model.generate_content(prompt)
        meal_data = json.loads(response.text.strip())

        # Add safety defaults if anything is missing
        meal_data.setdefault("meal_name", request.food_description)
        meal_data.setdefault("calories", 0)
        meal_data.setdefault("protein", 0)
        meal_data.setdefault("fat", 0)
        meal_data.setdefault("carbs", 0)

        date_str = str(request.date)

        log = nutrition_logs_collection.find_one({"uid": request.uid, "date": date_str})

        if log:
            log["meals"].append(meal_data)
            log["total"]["calories"] += meal_data["calories"]
            log["total"]["protein"] += meal_data["protein"]
            log["total"]["fat"] += meal_data["fat"]
            log["total"]["carbs"] += meal_data["carbs"]
            nutrition_logs_collection.update_one(
                {"uid": request.uid, "date": date_str},
                {"$set": {
                    "meals": log["meals"],
                    "total": log["total"]
                }}
            )
        else:
            nutrition_logs_collection.insert_one({
                "uid": request.uid,
                "date": date_str,
                "meals": [meal_data],
                "total": {
                    "calories": meal_data["calories"],
                    "protein": meal_data["protein"],
                    "fat": meal_data["fat"],
                    "carbs": meal_data["carbs"]
                }
            })

        return {"message": "Custom meal logged successfully", "data": meal_data}

    except Exception as e:
        print("Gemini error:", e)
        raise HTTPException(status_code=500, detail="Failed to estimate nutrition")
