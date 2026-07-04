from dataset.loader import DatasetLoader

loader = DatasetLoader("tests/sample.txt")

data = loader.load()

print(type(data))

print(data)