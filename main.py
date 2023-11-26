
from hwe import NewsPapersExtractor
from hwe.utils import jp2_to_jpg, noise_reduction, binarization
import pytesseract


if __name__ == '__main__':
    # NewsPapersExtractor('sn83030313').pages(limit=50, save=True)

    # convert jp2 to jpg
    # jp2_to_jpg('The New York herald. [volume]/1842-01-06-seq-1.jp2', 'The New York herald. [volume]/1842-01-06-seq-1.jpg')
    # noise_reduction('The New York herald. [volume]/1842-01-06-seq-1.jpg')
    # binarization('The New York herald. [volume]/1842-01-06-seq-1.jpg')
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    print(pytesseract.image_to_string('The New York herald. [volume]/1842-01-06-seq-1.jpg-binarized.jpg'))