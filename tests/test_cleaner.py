from dataset.cleaner import DatasetCleaner

text = """
<h1>Hello Bro</h1>

Visit : https://aira.ai

Email : test@gmail.com

This      is         AIRA.


"""

cleaner = DatasetCleaner()

clean = cleaner.clean(text)

print(clean)