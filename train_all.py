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