#text preprocessing with spacy

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from collections import defaultdict

nlp = spacy.blank('en')
data = pd.read_json("restaurant.json")
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
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
matcher.add("MENU", menu_token_items)

for idx, review in data.iterrows():
    doc = nlp(review.text)
    matches = matcher(doc)
    found_items = set([doc[match[1]:match[2]].text.lower() for match in matches])
    for item in found_items:
        item_ratings[item].append(review.stars)

mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}
#print(mean_ratings)

#find the best and the worst menu
sorted_menu = sorted(mean_ratings, key=mean_ratings.get)
print(f"Best Menu: {sorted_menu[-1]}, Worst Menu: {sorted_menu[0]}")









