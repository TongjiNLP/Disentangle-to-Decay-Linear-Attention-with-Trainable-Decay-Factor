from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model.LinearEluWithD2D import LinearEluWithD2DLMHeadModel


def generate_text(prompt, checkpoint_path = "" ,model_name='gpt2', max_length=200):
    """
    Generate text using a GPT model.

    :param prompt: The initial text to start the generation.
    :param model_name: The name of the model to use.
    :param max_length: The maximum length of the sequence to generate.
    :return: The generated text.
    """
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = LinearEluWithD2DLMHeadModel.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Encode the prompt text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length)

    # Generate text
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    prompt = "How are you?"
    text = generate_text(prompt)
    print(text)
