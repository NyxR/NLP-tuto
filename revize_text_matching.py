import pandas as pd
import spacy
from collections import defaultdict
from spacy.matcher import PhraseMatcher

data = pd.read_json("restaurant.json")
nlp = spacy.load("en_core_web_sm")

menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese",
        "Italian Combo", "Tiramisu", "Cannoli", "Chicken Salad",
        "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",
        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  
         "Prosciutto", "Salami"]

menu_items = [nlp(item) for item in menu]
item_ratings = defaultdict(list)
matcher = PhraseMatcher()
matcher.add("MENU", menu_items)

for index, review in data.iterrows():
    doc = nlp(review.text)
    matches = matcher(doc)
    found_items = set([doc[match[1]:match[2]].text.lower() for match in matches])
    for item in found_items:
        item_ratings[item].append(review.stars)

mean_ratings = {item: sum(rating)/len(rating) for item, rating in item_ratings.items}
sorted_rates = sorted(mean_ratings, key=mean_ratings.get)
best_menu = sorted_rates[-1]
worst_menu = sorted_rates[0]


