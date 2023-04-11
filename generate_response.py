import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


def generate_text(prompt, model, tokenizer, max_length=70):
    encoded_prompt = tokenizer.encode(prompt, return_tensors="tf")
    input_len = tf.shape(encoded_prompt)[1]
    attention_mask = tf.ones((1, input_len))

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=0.5,
        top_k=0,
        top_p=0.9,
    )

    generated_text = tokenizer.decode(
        output_sequences[0], clean_up_tokenization_spaces=True
    )
    return generated_text


def main():
    model_name = "./fine_tuned_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained(model_name)

    prompt = ""

    generated_text = generate_text(prompt, model, tokenizer)
    print(generated_text)


if __name__ == "__main__":
    main()
