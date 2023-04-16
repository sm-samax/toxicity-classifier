import nltk
import string
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
  
nltk.download('stopwords')
nltk.download('wordnet')

class StringStandardizer():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = WhitespaceTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def __punctuation_removal(self, input_data):
        return input_data.translate(str.maketrans('', '', string.punctuation))

    def __lemmatizing(self, input_data):
        lowercase = input_data.lower()
        raw_tokens = self.tokenizer.tokenize(lowercase)
        filtered_input_data = []
        for word in raw_tokens:
            lemmatized_word = self.lemmatizer.lemmatize(word)
            filtered_input_data.append(lemmatized_word)
        return filtered_input_data

    def __stop_word_removal(self, input_data):
        tokens = []
        for raw_token in input_data:
            if raw_token not in self.stop_words:
                tokens.append(raw_token)
        return tokens

    def standardize(self, str_arr):
        output_arr = []
        for text in str_arr:
            text = self.__punctuation_removal(text)
            text = self.__lemmatizing(text)
            text = self.__stop_word_removal(text)
            text = ' '.join(text)
            output_arr.append(text)
        return output_arr
pass
