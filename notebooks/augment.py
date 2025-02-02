# %%
import nltk
from nltk.corpus import wordnet
import random

# nltk.download('wordnet')
# nltk.download('omw-1.4')

def synonym_augmentation(sentence, num_replacements=1):
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

sentence = "The quick brown fox jumps over the lazy dog."
print(synonym_augmentation(sentence, num_replacements=2))

# %%
def random_insertion(sentence, n=2):
    words = sentence.split()
    for _ in range(n):
        new_word = random.choice(words)
        position = random.randint(0, len(words))
        words.insert(position, new_word)
    return ' '.join(words)

sentence = "The quick brown fox jumps over the lazy dog."
print(random_insertion(sentence, n=2))

# %%
def random_deletion(sentence, p=0.2):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return ' '.join(new_words) if new_words else random.choice(words)

sentence = "The quick brown fox jumps over the lazy dog."
print(random_deletion(sentence, p=0.2))

# %%
from googletrans import Translator

def back_translation(sentence, src_lang='en', mid_lang='fr'):
    translator = Translator()
    translated = translator.translate(sentence, src=src_lang, dest=mid_lang).text
    back_translated = translator.translate(translated, src=mid_lang, dest=src_lang).text
    return back_translated

sentence = "The quick brown fox jumps over the lazy dog."
print(back_translation(sentence))
