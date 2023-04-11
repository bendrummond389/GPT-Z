import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split

def load_dataset(file_path, tokenizer, block_size=128, test_size=0.2):
    df = pd.read_csv(
        file_path, sep="</s>", names=["prompt", "response"], engine="python"
    )

    # Remove any rows with missing values
    df = df.dropna()

    # Combine the prompts and responses into a single text
    text = ""
    for index, row in df.iterrows():
        if row["prompt"] is not None and row["response"] is not None:
            text += row["prompt"] + " " + row["response"] + " "

    tokenized_text = tokenizer.encode(text)

    num_blocks = len(tokenized_text) // block_size
    tokenized_text = tokenized_text[: num_blocks * block_size]

    tokenized_text = np.array(tokenized_text).reshape(num_blocks, block_size)

    # Shift the input one position to the right to create the labels
    input_text = tokenized_text[:, :-1]
    label_text = tokenized_text[:, 1:]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        input_text, label_text, test_size=test_size, random_state=42
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(4)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(4)

    return train_dataset, test_dataset


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

train_dataset, test_dataset = load_dataset("processed_data.txt", tokenizer)


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)


# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2")

# Print the training history
print("Training history:", history.history)
