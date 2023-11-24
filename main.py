
from hwe import NewsPapersExtractor

if __name__ == '__main__':
    print(NewsPapersExtractor('sn83030313').pages(limit=50, save=True))