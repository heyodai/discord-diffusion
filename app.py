from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import discord
import os

TOKEN = os.environ['GITHUB_BOT_TOKEN']
ALLOW_NSFW = True

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def main():
    pipe = init_pipe()
    tokenizer, model = init_magic_prompt()

    @client.event
    async def on_ready():
        print(f'We have logged in as {client.user}')

    @client.event
    async def on_message(message):
        if message.author == client.user:
            # don't respond to ourselves
            return

        if message.content.startswith('$art'):
            # get the message after the command
            msg = message.content[5:]
            # await message.channel.send(f'Hello {msg}!')

            # embed a local image (test.png)
            embed = discord.Embed()
            with open('test.png', 'rb') as f:
                picture = discord.File(f)
                embed.set_image(url='attachment://test.png')
                await message.channel.send(file=picture, embed=embed, content=f'Hello {msg}!')

        if message.content.startswith('$prompt'):
            msg = message.content[8:]
            if msg == '':
                await message.channel.send(f'Please provide a prompt.')
                return

            variation = enhance_prompt(msg, tokenizer, model)
            await message.channel.send(f'Here are some variations on your prompt:) {variation}')

    client.run(TOKEN)

def init_pipe():
    """
    Create a new pipeline and configure it for use with the Stable Diffusion model.

    @return StableDiffusionPipeline
    """
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("mps")

    # Toggle the NSFW filter
    #
    # The filter is intended to prevent the model from generating images that are
    # inappropriate for the general public. However, it is not perfect and can
    # sometimes block images that are not NSFW.
    #
    # @see https://github.com/CompVis/stable-diffusion/issues/239
    if ALLOW_NSFW:
        pipe.safety_checker = lambda images, **kwargs: (images, False)

    # Enable sliced attention computation.
    #
    # When this option is enabled, the attention module will split the input tensor in slices, 
    # to compute attention in several steps. This is useful to save some memory in exchange for 
    # a small speed decrease.
    #
    # Per Hugging Face, recommended if your computer has < 64 GB of RAM.
    pipe.enable_attention_slicing()

    return pipe

def init_magic_prompt():
    """
    Initialize the tokenizer and model for Magic Prompt.

    @return tuple(AutoTokenizer, AutoModelForCausalLM)
    """
    tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
    model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion") 
    return tokenizer, model

def enhance_prompt(prompt, tokenizer, model):
    """
    Generate prompt variations using Magic Prompt.

    @param prompt string
    @param tokenizer AutoTokenizer
    @param model AutoModelForCausalLM
    @return string
    """
    # tokenize the prompt
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")

    # generate the variations
    variations = model.generate(
        tokenized_prompt,
        do_sample=True,
        max_length=500,
        top_k=100,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=4,
        attention_mask=None,
    )

    # decode the variations
    variations = tokenizer.batch_decode(variations, skip_special_tokens=True)

    # return the first variation
    return variations[0]

def make_image():
    start_time = time.time()

if __name__ == '__main__':
    main()