import pickle
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import streamlit as st
st.title("Sentiment Analysis")
st.markdown("Overview: Takes the tweet as input, analyzes it and classifies the sentiment of the tweet as positive,neutral or negative.")
abc=st.text_input("Tweet your mood below")
if st.button("Analyze"):
    st.text("Analyzing may take upto a minute. Please be patient. Thank you!")
    df = pickle.load(open("df.pkl","rb"))
    tokenizer = Tokenizer()
    docs=df["selected_text"].astype("string")
    tokenizer.fit_on_texts(docs)
    sequences = tokenizer.texts_to_sequences(docs)
    sequences = pad_sequences(sequences,padding='post',maxlen=22)
    voc_size=len(tokenizer.word_index)
    model = Sequential()
    model.add(Embedding(voc_size+1,2,input_length=22))
    model.add(SimpleRNN(32,return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    X=sequences
    Y=df['sentiment']
    Y=Y.to_numpy()
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['acc'])
    model.fit(X,Y,epochs=5)
    abc=[abc]
    seq=tokenizer.texts_to_sequences(abc)
    inp=pad_sequences(seq,padding='post',maxlen=22)
    a=model.predict(inp)
    value=a.argmax()
    if value == 0:
        st.text("Result: Negative mood")
    elif value == 1:
        st.text("Result: Neutral mood")
    else:
        st.text("Result: Positive mood")
