import ee
import datetime

PROJECT="river-key-323305"

ee.Initialize(project=PROJECT)

collection=ee.ImageCollection(
    "GOOGLE/Research/open-buildings-temporal/v1"
)

dates=collection.aggregate_array("system:time_start").getInfo()

years=sorted(set(
    datetime.datetime.utcfromtimestamp(t/1000).year
    for t in dates
))

print(years)