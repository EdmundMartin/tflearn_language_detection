# tflearn_language_detection
language_detect.py provides a model which allows users to train language dectection classifiers using text files. The model makes use of a TFLearn LTSM model and allows for highly accurate language detection. 

## Training a Model
The train_all.py provides an example of how you can train a model. The file iterates through all the text files contained in the language, extracting the label from the file name before loading them into our model. Once all the text files have been loaded we then train the langauage detection classifier for one epoch. We can then save the model and associated vocabs' using Python's pickle.
```python
from language_detect import LanguageDetection
import glob

language_files = glob.glob('language_texts/*.txt')

Ld = LanguageDetection(vocab_size=5000, test_split=0.3)
for filename in language_files:
    lang_name = filename.split('\\')
    lang_name = lang_name[1].split('.')[0]
    Ld.load_words(filename, lang_name)
Ld.train_model(epochs=1, batch_size=16)
Ld.save_model('all_languages')
```
