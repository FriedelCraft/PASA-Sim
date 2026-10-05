import ee

PROJECT = "river-key-323305"

try:
    ee.Initialize(project=PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT)

print("Earth Engine initialized successfully!")

DATASET = "GOOGLE/Research/open-buildings-temporal/v1"

try:
    collection = ee.ImageCollection(DATASET)

    print("Dataset exists.")
    print()

    print("Number of images:")
    print(collection.size().getInfo())

    first = collection.first()

    print("\nFirst image bands:")
    print(first.bandNames().getInfo())

    print("\nProjection:")
    print(first.projection().getInfo())

    print("\nProperties:")
    print(first.propertyNames().getInfo())

except Exception as e:
    print("ERROR:")
    print(e)