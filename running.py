# Load the model
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr


class args:
    model_name_or_path = "./guanaco_all_1_3b"  # Choose your model here
    use_fast_tokenizer = True


tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    use_fast=args.use_fast_tokenizer,
    use_auth_token=None,
)

# Load the trained model
pt_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)


# function for gradio
def response(message, history=None):
    input_text = message
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    output = pt_model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        top_k=30,
        top_p=0.95,
        repetition_penalty=1.2,
        num_return_sequences=1
    )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return response_text


# Launch the interface
iface = gr.ChatInterface(
    fn=response,
)
iface.launch(share=True)
