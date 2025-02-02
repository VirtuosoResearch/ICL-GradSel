# %%
import nltk
from nltk.corpus import wordnet
import random

from deep_translator import GoogleTranslator

# nltk.download('wordnet')
# nltk.download('omw-1.4')
class DataAugment():
    def __init__(self):
        pass

    def synonym_augmentation(self, sentence, num_replacements=1):
        words = sentence.split()
        augmented_sentence = words.copy()
        for _ in range(num_replacements):
            word_to_replace = random.choice(words)
            synonyms = wordnet.synsets(word_to_replace)
            if synonyms:
                synonym_words = [lemma.name() for syn in synonyms for lemma in syn.lemmas()]
                if synonym_words:
                    synonym = random.choice(synonym_words)
                    augmented_sentence[words.index(word_to_replace)] = synonym.replace("_", " ")
        return " ".join(augmented_sentence)

    def random_insertion(self, sentence, n=2):
        words = sentence.split()
        for _ in range(n):
            new_word = random.choice(words)
            position = random.randint(0, len(words))
            words.insert(position, new_word)
        return ' '.join(words)

    def random_deletion(self, sentence, p=0.2):
        words = sentence.split()
        if len(words) == 1:
            return sentence
        new_words = [word for word in words if random.uniform(0, 1) > p]
        return ' '.join(new_words) if new_words else random.choice(words)

    def back_translation(self, sentence, src_lang='en', mid_langs=['fr', 'es', 'de']):
        mid_lang = random.choice(mid_langs) 
        mid_translation = GoogleTranslator(source=src_lang, target=mid_lang).translate(sentence)
        back_translated = GoogleTranslator(source=mid_lang, target=src_lang).translate(mid_translation)
        return back_translated

sentence = "DeepSeek defeat OpenAI in the latest scientific contest between US and PRC."

augment = DataAugment()

print(augment.synonym_augmentation(sentence))
print(augment.random_insertion(sentence))
print(augment.back_translation(sentence))
print(augment.random_deletion(sentence))
