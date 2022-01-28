import re

s = "Welcome to GeeksForGeeks"

res = re.search(r'(\D{2})? t', s)

print(res.group(1))