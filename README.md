COMPANY:CODTECH IT SOLUTIONS 
NAME:ABDUL RAZZAK SHEIKH 
INTERNS J20: COD01234 
DOMAIN:ARTIFICIAL INTELLIGENCE 
DURATION:4 WEEKS 
MENTOR: NEELA SANTOSH 
##DESCRIPTION:i have used openai module which provides access to OpenAI’s GPT models,tensorflow module which is required for deep learning tasks,tokenizer module which converts text into numerical sequences,pad_sequences which ensures sequences have equal length,Sequential module which defines a neural network model,LSTM module which is a recurrent neural network layer used for text generation,Embedding module which Converts words into dense vector representations,Dense module which is a fully connected layer for output predictions and numpy module which is used for numerical operations.
def generate_text_gpt(prompt, api_key, model="gpt-4", max_tokens=200):function uses OpenAI’s GPT-4 to generate text.
Takes prompt (text input), API key, model name (gpt-4), and max_tokens (limits response length).
openai.api_key = api_key: Sets the API key.
Calls OpenAI’s API using openai.ChatCompletion.create().
model=model: Specifies the GPT model.
messages: Defines a conversation history:
"role": "system": Sets the assistant’s behavior.
"role": "user": Contains the user's input.
max_tokens: Limits output length.
Extracts and returns the generated text.
It Uses GPT-4 for advanced text generation via OpenAI’s API.
and Implements an LSTM-based model for training on small datasets and
Demonstrates tokenization, sequence padding, and text generation.
def train_lstm_model(text_data, vocab_size=10000, embedding_dim=64, lstm_units=128, max_length=50):
in this function Tokenization
Tokenizer(num_words=vocab_size, oov_token="<OOV>"):
Creates a tokenizer that keeps the vocab_size most frequent words.
oov_token="<OOV>" replaces unknown words with <OOV>.
tokenizer.fit_on_texts(text_data):
Builds a word-to-index dictionary from text_data.
sequences = tokenizer.texts_to_sequences(text_data):
Converts text samples into sequences of integers.
pad_sequences(sequences, maxlen=max_length, padding="post"):
Ensures all sequences have equal length (max_length).
padding="post": Pads shorter sequences at the end.
LSTM Model Definition 
Sequential([]): Creates a sequential neural network.
Embedding Layer (Embedding(vocab_size, embedding_dim, input_length=max_length)):
Maps words to dense vectors of size embedding_dim.
Helps the LSTM layer understand word relationships.
First LSTM Layer (LSTM(lstm_units, return_sequences=True)):
Processes word sequences.
return_sequences=True: Keeps full sequences for the next LSTM layer.
Second LSTM Layer (LSTM(lstm_units)):
Processes the final sequence into a single vector.
Dense Output Layer (Dense(vocab_size, activation="softmax")):
Outputs a probability distribution over the vocabulary.
Compiling & Displaying the Model
loss="sparse_categorical_crossentropy":
Used for multi-class classification.
optimizer="adam":
Adaptive learning rate optimization.
metrics=["accuracy"]:
Tracks training accuracy.
model.summary():
Displays model architecture.
Returns the trained model and tokenizer.
def generate_text_lstm(model, tokenizer, seed_text, max_length=50, num_words=50):
in this function 
seed_text: The starting text for generation.
Loops num_words times, predicting the next word each iteration.
Text Processing:
tokenizer.texts_to_sequences([result]): Converts text to a sequence.
pad_sequences(sequence, maxlen=max_length, padding="post"): Pads the sequence.
Predicting Next Word:
model.predict(padded_sequence): Gets the model’s word probabilities.
np.argmax(..., axis=-1)[0]: Finds the word index with the highest probability.
tokenizer.index_word.get(predicted_index, ""): Converts index to word.
Appending the Predicted Word:
result += " " + predicted_word
Returns the final generated text.
Running the Script
Checks if the script is run as the main program (__name__ == "__main__").
GPT-based text generation (commented out; requires an API key).
LSTM Model Training:
Defines a small dataset (text_samples).
Calls train_lstm_model() to train the LSTM model.
Calls generate_text_lstm() to generate text using the trained model.
Conclusion
Uses GPT-4 for advanced text generation via OpenAI’s API.
Implements an LSTM-based model for training on small datasets.
Demonstrates tokenization, sequence padding, and text generation.
