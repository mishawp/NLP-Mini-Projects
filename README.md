# 🧠 NLP Mini Projects

Репозиторий содержит 10 учебных мини-проектов по обработке естественного языка (NLP), выполненных во время прохождения курса [**Natural Language Processing Demystified**](https://www.nlpdemystified.org/). Проекты охватывают ключевые методы и архитектуры — от классических статистических моделей до современных нейросетевых подходов на основе Transformer.

## ⚙️ Стек технологий

- `spaCy` (01, 02, 03, 06)
- `gensim` (03, 04, 05)
- `scikit-learn` (01)
- `torch` (02, 05, 06, 07, 08, 09, 10)
- `pytorch-lightning` (08, 09, 10)
- `tokenizers` (07, 08, 09)
- `transformers` (10)

## 💡 Что демонстрирует этот репозиторий

- Опыт работы с вышеперечисленными библиотеками  
- Владение базовыми и современными методами NLP: `Bag of Words`, `TF-IDF`, `Naive Bayes`, `LDA`, `Neural Networks`, `Embeddings`, `LSTM`, `Seq2Seq`, `Attention`, `Transformer`  
- Структурированное оформление проектов и аккуратные ноутбуки

## 📚 Проекты

### [01. Classification — Naive Bayes](01.Classification.Naive-Bayes.ipynb)

**Задача:** Классификация текстов  
**Датасет:** [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)  
**Токенизация:** `spaCy` (уровень слов)  
**Векторизация:** `TfidfVectorizer`, `CountVectorizer`  
**Модель:** `MultinomialNB` (наивный байесовский классификатор)

---

### [02. Classification — Neural Network](02.Classification.Neural-Network.ipynb)

**Задача:** Классификация текстов  
**Датасет:** [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)  
**Токенизация:** `spaCy`  
**Векторизация:** `TfidfVectorizer`  
**Модель:** Полносвязная нейронная сеть

---

### [03. Topic Modeling — LDA](03.Topic-Modeling.LDA.ipynb)

**Задача:** Тематическое моделирование  
**Датасет:** CNN Articles (≈90 000 новостей)  
**Токенизация:** `spaCy`  
**Векторизация:** `gensim.corpora.Dictionary` (`.doc2bow`)  
**Модель:** `LdaModel` (*Latent Dirichlet Allocation*)

---

### [04. Word Vectors — gensim](04.Review.Word-Vectors-gensim.ipynb)

**Задача:** Работа с векторными представлениями слов (*Word Embeddings*)  
**Модель:** Предобученные эмбеддинги на корпусе *Google News (2015)*  
**Цель:** Исследование семантических связей между словами

---

### [05. Sentiment Analysis — Neural Network](05.Sentiment-Analysis.Neural-Network.ipynb)

**Задача:** Анализ тональности текстов  
**Датасет:** Yelp Polarity Reviews (Amazon)  
**Токенизация:** Кастомный `WordLevel`  
**Векторизация:** Embedding слой (с предобученными эмбеддингами)  
**Модель:** Полносвязная нейронная сеть

---

### [06. Part-of-Speech Tagging — BiLSTM](06.Part-of-Speech-Tagging.Bidirectional-LSTM.ipynb)

**Задача:** Разметка частей речи  
**Датасет:** Корпусы `nltk`: `treebank`, `brown`, `conll2000`  
**Токенизация:** Кастомный `WordLevel`  
**Векторизация:** Embedding слой  
**Модель:** Bidirectional LSTM

---

### [07. Language Modeling — LSTM](07.Language-Modelling.LSTM.ipynb)

**Задача:** Моделирование языка  
**Датасет:** *The Art of War* (Сунь-цзы)  
**Токенизация:** `tokenizers.Tokenizer` (`WordLevel`)  
**Векторизация:** One-Hot Encoding  
**Модель:** Двухслойная LSTM

---

### [08. Machine Translation — Seq2Seq + Attention](08.Machine-Translation.LSTM-Seq2Seq-Attention.ipynb)

**Задача:** Машинный перевод (русский → английский)  
**Датасет:** [Tatoeba](https://tatoeba.org/en/downloads)  
**Токенизация:** `tokenizers.Tokenizer` (`WordLevel`)  
**Векторизация:** Embedding слой  
**Модель:** Seq2Seq LSTM с/без Attention

---

### [09. Machine Translation — Transformer from Scratch](09.Machine-Translation.Transformer-from-Scratch.ipynb)

**Задача:** Машинный перевод (русский → английский)  
**Датасет:** [Tatoeba](https://tatoeba.org/en/downloads)  
**Токенизация:** `tokenizers.Tokenizer` (`WordLevel` / `BPE`)  
**Векторизация:** Embedding слой  
**Модель:** Оригинальный Transformer, реализованный с нуля

---

### [10. Question Answering — distilRoBERTa](10.Question-Answering.Fine-Tuned-distilroberta.ipynb)

**Задача:** Извлечение ответа на вопрос из контекста (Question Answering)  
**Датасет:** *SQuAD* (`Hugging Face datasets`)  
**Токенизация:** `distilroberta-base` `AutoTokenizer`  
**Модель:** `distilroberta-base`
