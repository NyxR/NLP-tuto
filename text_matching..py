import pandas as pd
import numpy as np
import spacy
from spacy import matcher
from spacy.matcher import PhraseMatcher
from collections import defaultdict

nlp = spacy.load('en_core_web_sm')
review_data = pd.read_json('restaurant.json')
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
item_ratings = defaultdict(list)
menu_token_items = [nlp(item) for item in menu]
#define the matcher
matchers = PhraseMatcher(nlp.vocab, attr='LOWER')
matchers.add("MENU", menu_token_items)

for index, review in review_data.iterrows():
    doc = nlp(review.text)
    matches = matchers(doc)
    found_items = set([doc[match[1]:match[2]].text.lower() for match in matches])
    for item in found_items:
        item_ratings[item].append(review.stars)

mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}
#The Best and the worst menu:
sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)
best_menu = sorted_ratings[-1]
worst_menu = sorted_ratings[0]
