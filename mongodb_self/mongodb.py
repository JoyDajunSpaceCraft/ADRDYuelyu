import json

from pymongo import MongoClient

CONNECTION_STRING = "mongodb://localhost:27017/primer"
client = MongoClient(CONNECTION_STRING)
db = client.primer
units_ = db.temperture
# 2 What is the winter (December to February) average temperature per city in all years?
# results = units_.aggregate([
# {"$match":{"Month":{"$in":["12","1","2"]}}},
#     {"$group":{"_id":{
#         "city":"$City",
#         "AverageTemperature":"$AverageTemperature"}}}
# ])

# 3 What is the average temperature per country every year?
# results = units_.aggregate([
#     { "$sort": { "AverageTemperature": -1 } },
#     {"$group":{
#         "_id":{"country":"$Country",
#         "year":"$Year",
#         "AverageTemperature":"$AverageTemperature"},
#     }}
# ])

# 4 What is the highest average temperature per city, and which year?
# results = units_.aggregate([
#     { "$sort": { "AverageTemperature": -1 } },
#     {"$group":{
#         "_id":{"city":"$City",
#         "year":"$Year",
#         "AverageTemperature":"$AverageTemperature"},
#     }}
# ])

# 5 What countries =Denmark, and in the year = 1745, have the highest average temperature?
# results = units_.aggregate([
#     {"$match": {"Country": "Denmark", "Year": "1745"}},
#     {"$sort": {
#         "AverageTemperature": -1
#     }}]
# )

# 6 What countries =Denmark and in year = 1745 all the spring (March to May) average temperature?
# results = units_.aggregate([
#     {"$match": {"Country": "Denmark",
#                 "Year": "1745",
#                 "Month":{ "$in": ["3","4","5"]}}},
#     {"$sort": {
#         "AverageTemperature": -1
#     }}]
# )

# 7 What is the average temperature uncertainty per country in all years?
# results = units_.aggregate([
#
#     {"$group": {
#             "_id":{"country":"$Country",
#             "year":"$Year",
#             "AverageTemperatureUncertainty":"$AverageTemperatureUncertainty"},
#         }}]
# )

# 7 What are the average temperature uncertainty latitude and longitude in the tropics (+-23.5 latitude) per year?
results = units_.aggregate([
    {"$match": {"Latitude": {"$lt": "23.5"}}},
    {"$limit":1000},
    {"$group": {
        "_id": {"country": "$Country",
                "AverageTemperatureUncertainty": "AverageTemperatureUncertainty",
                "year":"$Year",
                "Latitude": "$Latitude",
                "Longitude":"$Longitude"},
    }},
]
)

# 8 What is the average temperature per city in all years?
# results = units_.aggregate([
#     {"$group": {
#         "_id": {"city": "$City",
#                 "AverageTemperature": {"avg":"$AverageTemperature"}},
#     }}]
# )

# 9 What are the average temperature uncertainty latitude and longitude in Temperates (+-23.5- +-66.5 latitude) per year?
# results = units_.aggregate([
#     {"$match": {"Latitude": {"$gt": "23.5","$lt":"66.5"}}},
#     {"$limit":1000},
#     {"$group": {
#         "_id": {"city": "$City",
#                 "AverageTemperatureUncertainty": "$AverageTemperatureUncertainty",
#                 "Latitude": "$Latitude",
#                 "Longitude": "$Longitude"
#                 },
#     }}
# ])

# 10 What is the temperature and temperature uncertainty per year range 1900-2000?
# results = units_.aggregate([
#     {"$match": {"Year": {"$gt": "1900", "$lt": "2000"}}},
#     {"$group": {
#         "_id": {
#             "country": "$Country",
#             "AverageTemperatureUncertainty": "$AverageTemperatureUncertainty",
#             "AverageTemperature": "$AverageTemperature",
#         },
#     }}
# ]
# )

list_1 = []
for x in results:
    if x not in list_1:
        print(x)
        list_1.append(x)
# print(list_1)
with open("mongo_query7.json", "w", encoding="utf-8") as files:
    json.dump(list_1, files)
