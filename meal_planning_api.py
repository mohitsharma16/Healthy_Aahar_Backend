from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pymongo
import os
import re
import json
import time
from datetime import date, timedelta
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
recipe_feedback_collection = db["recipe_feedback"] 


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
        
        # # Try to extract text safely based on your SDK
        # response_text = getattr(response, "text", None)
        # if not response_text and hasattr(response, "candidates"):
        #     response_text = response.candidates[0].content.parts[0].text
        
        # if not response_text:
        #     raise ValueError("Empty or invalid Gemini response")
        response_text = response.text.strip()
        
        print("Raw Gemini response:")
        print(response_text)

        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match:
            raise ValueError("Could not extract JSON from Gemini response")
        recipe_dict = json.loads(match.group())
        result = recipes_collection.insert_one(recipe_dict)
        recipe_dict["_id"] = str(result.inserted_id)

        return recipe_dict
        # recipes_collection.insert_one(recipe_data)
        # return recipe_data
    except Exception as e:
        print(f"Error generating recipe with ingredients: {e}")
        return None


from fastapi.encoders import jsonable_encoder
from bson import ObjectId

from datetime import datetime

def custom_jsonable_encoder(obj):
    return jsonable_encoder(
        obj,
        custom_encoder={ObjectId: str}
    )

@app.get("/generate_meal_plan/{uid}")
def generate_meal_plan(uid: str):
    user = users_collection.find_one({"uid": uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_name = user["name"]
    today_date = datetime.utcnow().date()
    date_str = str(today_date)

    existing_plan = meal_plans_collection.find_one({
        "uid": uid,
        "date": date_str
    })

    if existing_plan:
        # Get today's logged meals to check which ones are already logged
        nutrition_log = nutrition_logs_collection.find_one({"uid": uid, "date": date_str})
        logged_meal_ids = []
        if nutrition_log and "meals" in nutrition_log:
            logged_meal_ids = [meal.get("meal_id", "") for meal in nutrition_log["meals"] if meal.get("meal_id")]

        # Update isLogged status based on what's actually logged
        for meal in existing_plan["meal_plan"]:
            meal_id = str(meal.get("_id", ""))
            meal["isLogged"] = meal_id in logged_meal_ids

        # Update the meal plan in database with correct isLogged status
        meal_plans_collection.update_one(
            {"uid": uid, "date": date_str},
            {"$set": {"meal_plan": existing_plan["meal_plan"]}}
        )

        return custom_jsonable_encoder(existing_plan)

    daily_calories = user["tdee"]
    calorie_target = daily_calories / 3
    goal = user["goal"]

    if goal == "weight_loss":
        calorie_target -= 100
    elif goal == "weight_gain":
        calorie_target += 100

    meal_plan = []
    meals_cursor = recipes_collection.find({
        "Calories": {"$gte": calorie_target - 50, "$lte": calorie_target + 50}
    })
    meals_list = list(meals_cursor)

    random.shuffle(meals_list)

    while len(meals_list) < 3:
        ai_recipe = generate_recipe(calorie_target, goal)
        if ai_recipe:
            inserted = recipes_collection.insert_one(ai_recipe)
            ai_recipe["_id"] = str(inserted.inserted_id)
            meals_list.append(ai_recipe)
        else:
            break

    if not meals_list:
        raise HTTPException(status_code=404, detail="No suitable meals found")

    # Get today's logged meals to check which ones are already logged
    nutrition_log = nutrition_logs_collection.find_one({"uid": uid, "date": date_str})
    logged_meal_ids = []
    if nutrition_log and "meals" in nutrition_log:
        logged_meal_ids = [meal.get("meal_id", "") for meal in nutrition_log["meals"] if meal.get("meal_id")]

    for _ in range(3):
        meal = meals_list.pop(0)
        if "_id" in meal:
            meal["_id"] = str(meal["_id"])
        
        # Check if this meal is already logged
        meal_id = str(meal.get("_id", ""))
        meal["isLogged"] = meal_id in logged_meal_ids
        
        meal_plan.append(meal)

    meal_plan_data = {
        "uid": uid,
        "user_name": user_name,
        "meal_plan": meal_plan,
        "date": date_str
    }

    meal_plans_collection.insert_one(jsonable_encoder(meal_plan_data))
    return jsonable_encoder(meal_plan_data)


class SwapMealRequest(BaseModel):
    meal_index: int

@app.put("/swap_meal/{uid}")
def swap_meal(uid: str, request: SwapMealRequest):
    user_meal_plan = meal_plans_collection.find_one({"uid": uid})
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
        meal_plans_collection.update_one({"uid": uid}, {"$set": {"meal_plan": meal_plan}})
        return jsonable_encoder({"message": "Meal swapped successfully", "new_meal": new_meal})

    return {"message": "No suitable meal found for swapping"}


class LogMealRequest(BaseModel):
    uid: str
    meal_id: str  # ID of the recipe from your recipe collection
    date: date    # Format: "YYYY-MM-DD"
    meal_type: str  # new field: "breakfast", "lunch", or "dinner"

@app.post("/log_meal")
def log_meal(request: LogMealRequest):
    user = users_collection.find_one({"uid": request.uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        print("meal_id received:", request.meal_id)
        print("Type of meal_id:", type(request.meal_id))
        recipe = recipes_collection.find_one({"_id": ObjectId(request.meal_id.strip())})
        
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid meal_id format")
        
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    meal_data = {
        "meal_id": str(recipe["_id"]),
        "meal_name": recipe["TranslatedRecipeName"],
        "meal_type": request.meal_type,
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

    # UPDATE THE MEAL PLAN TO SET isLogged = True for this meal
    meal_plan = meal_plans_collection.find_one({"uid": request.uid, "date": date_str})
    if meal_plan and "meal_plan" in meal_plan:
        # Find the meal in the meal plan and update its isLogged status
        updated = False
        for i, meal in enumerate(meal_plan["meal_plan"]):
            meal_id_str = str(meal.get("_id", ""))
            if meal_id_str == request.meal_id:
                meal_plan["meal_plan"][i]["isLogged"] = True
                updated = True
                break
        
        # Update the meal plan in the database only if we found and updated the meal
        if updated:
            meal_plans_collection.update_one(
                {"uid": request.uid, "date": date_str},
                {"$set": {"meal_plan": meal_plan["meal_plan"]}}
            )

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

from fastapi import Body
import re

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
        response_text = response.text.strip()

        print("Raw Gemini response:")
        print(response_text)

        # Clean up response and parse as JSON
        json_str_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_str_match:
            raise ValueError("Gemini response doesn't contain valid JSON")

        json_str = json_str_match.group(0)
        meal_data = json.loads(json_str)

        # Add fallback values if keys are missing
        meal_data.setdefault("meal_name", request.food_description)
        meal_data.setdefault("calories", 0)
        meal_data.setdefault("protein", 0)
        meal_data.setdefault("fat", 0)
        meal_data.setdefault("carbs", 0)

        # Date string formatting
        date_str = str(request.date)

        # Fetch existing log
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

from fastapi import Query
from typing import Optional

#endpoint for meal history from fastapi import Query

@app.get("/get_meal_history")
def get_meal_history(uid: str = Query(...), date: Optional[str] = Query(None)):
    # Define query based on the uid and optional date
    query = {"uid": uid}
    if date:
        query["date"] = date  # Filter by date if provided

    # Fetch meals logged for the given user
    logs_cursor = nutrition_logs_collection.find(query)
    logs = list(logs_cursor)

    if not logs:
        raise HTTPException(status_code=404, detail="No meal history found for this user")

    # Convert ObjectId to string for compatibility with JSON
    for log in logs:
        log["_id"] = str(log["_id"])

    return jsonable_encoder(logs)

#endpoint for getting weekly reports@app.get("/weekly_report/{uid}")
@app.get("/weekly_report/{uid}")
def get_weekly_report(uid: str):
    user = users_collection.find_one({"uid": uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate the start and end date of the week
    today_date = datetime.utcnow().date()
    start_date = today_date - timedelta(days=today_date.weekday())  # Start of the week
    end_date = start_date + timedelta(days=6)  # End of the week

    # Fetch the meals logged during the week
    meal_history = nutrition_logs_collection.find({
        "uid": uid,
        "timestamp": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
    })

    meals = list(meal_history)

    if not meals:
        return {"message": "No meals logged this week"}

    # Summarize or return detailed data
    total_calories = sum(meal["calories"] for meal in meals)
    meal_count = len(meals)
    
    weekly_report = {
        "user_id": uid,
        "week_start": str(start_date),
        "week_end": str(end_date),
        "total_calories": total_calories,
        "meal_count": meal_count,
        "meals": meals
    }

    return jsonable_encoder(weekly_report)

#endpoint for user feedback

class RecipeFeedbackRequest(BaseModel):
    recipe_id: str
    rating: int  # Assuming rating is between 1 and 5
    comments: str

@app.post("/recipe_feedback/{uid}")
def recipe_feedback(uid: str, request: RecipeFeedbackRequest):
    user = users_collection.find_one({"uid": uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    feedback_data = {
        "uid": uid,
        "recipe_id": request.recipe_id,
        "rating": request.rating,
        "comments": request.comments,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Store the feedback
    recipe_feedback_collection.insert_one(feedback_data)
    return {"message": "Feedback submitted successfully"}


# recipe fetching and details endpoint 

@app.get("/get_recipe/{recipe_id}")
def get_recipe_details(recipe_id: str):
    """
    Get detailed information about a specific recipe by its ID
    """
    try:
        recipe = recipes_collection.find_one({"_id": ObjectId(recipe_id.strip())})
        
        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")
        
        # Convert ObjectId to string for JSON serialization
        recipe["_id"] = str(recipe["_id"])
        
        # Get average rating and feedback count for this recipe
        feedback_cursor = recipe_feedback_collection.find({"recipe_id": recipe_id})
        feedback_list = list(feedback_cursor)
        
        if feedback_list:
            total_ratings = sum(feedback["rating"] for feedback in feedback_list)
            avg_rating = round(total_ratings / len(feedback_list), 1)
            feedback_count = len(feedback_list)
        else:
            avg_rating = 0
            feedback_count = 0
        
        # Add rating information to recipe response
        recipe["average_rating"] = avg_rating
        recipe["feedback_count"] = feedback_count
        
        return jsonable_encoder(recipe)
        
    except Exception as e:
        # Handle invalid ObjectId format
        if "invalid ObjectId" in str(e).lower() or "not a valid ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid recipe ID format")
        else:
            print(f"Error fetching recipe details: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")


