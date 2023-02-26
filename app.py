from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import discord
import os
import pickle

TOKEN = os.environ['BOT_TOKEN']
STATUS_CHANNEL = os.environ['STATUS_CHANNEL_ID']
ALLOW_NSFW = True
NUM_INFERENCE_STEPS = 1
HEIGHT = 512
WIDTH = 512

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def main():
    pipe = init_pipe()
    tokenizer, model = init_magic_prompt()

    @client.event
    async def on_ready():
        print(f'We have logged in as {client.user}')

        # TODO: send a logoff message when the bot is shut down

        # Let's send a message to the status channel to let everyone know the bot is online
        # we'll also send a gif of Bender to celebrate
        bender = open('gifs/bender.gif', 'rb')
        bender = discord.File(bender, filename='bender.gif')

        embed = discord.Embed()
        embed.set_image(url='attachment://bender.gif')

        channel = client.get_channel(int(STATUS_CHANNEL))
        await channel.send(content='Bot is online.', file=bender, embed=embed)


    @client.event
    async def on_message(message):
        if message.author == client.user:
            # don't respond to ourselves
            return

        if message.content.startswith('$ping'):
            await message.channel.send('🏓 Pong! {0.author.mention}'.format(message))

        if message.content.startswith('$help'):
            msg = """
            Available commands:
            - $ping - Check if the bot is online
            - $help - Show this message
            - $draw [prompt] - Generate an image
            - $redraw [prompt] - Generate a variation of last image
            - $prompt [prompt] - Use Magic Prompt to generate a variation of the prompt
            - $wait - See current wait time
            """
            await message.channel.send(msg)
        
        if message.content.startswith('$draw') or message.content.startswith('$redraw'):
            # Get the prompt from the message
            is_redraw = message.content.startswith('$redraw')
            msg = message.content[6:] if not is_redraw else message.content[8:]

            # Guard clauses for bad input
            if msg == '' and not is_redraw:
                await message.channel.send(f'Please provide a prompt.')
                return
            if msg != '' and is_redraw:
                await message.channel.send(f'Please do not provide a prompt when using $redraw.')
                return
            
            # pickle the prompt for use with the $redraw command
            with open('prompt.pickle', 'wb') as f:
                pickle.dump(msg, f)
            if is_redraw:
                with open('prompt.pickle', 'rb') as f:
                    msg = pickle.load(f)

            # TODO: send estimated time to generate
            filename, time_elapsed = make_image(pipe, msg)
            image = open(f'output/{filename}', 'rb')
            image = discord.File(image, filename=f'output/{filename}')

            embed = discord.Embed()
            embed.set_image(url=f'attachment://{filename}')

            await message.channel.send(file=image, embed=embed, content=f'Time to generate: {time_elapsed}')
                
        if message.content.startswith('$prompt'):
            msg = message.content[8:]
            if msg == '':
                await message.channel.send(f'Please provide a prompt.')
                return

            variation = enhance_prompt(msg, tokenizer, model)
            await message.channel.send(f'New prompt: {variation}')

        if message.content.startswith('$wait'):
            await message.channel.send(f'¯\_(ツ)_/¯')

    @client.event
    async def on_disconnect():
        print('Bot disconnected.')

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

def make_image(pipe, prompt):
    """
    Generate an image using the Stable Diffusion model.

    @param pipe StableDiffusionPipeline
    @param prompt string
    @return tuple(string, string)
    """
    start_time = time.time()

    # Empty the output directory
    #
    # We do this to prevent the script from running out of disk space.
    for file in os.listdir('output'):
        os.remove(f'output/{file}')

    # First-time "warmup" pass
    #
    # This is necessary to get the model to run on Apple Silicon (M1/M2). There is a
    # bug in the MPS implementation of the model that causes it to crash on the first
    # pass. This is a workaround to get the model to run on Apple Silicon.
    #
    # It takes about 30 seconds to run.
    _ = pipe(prompt, num_inference_steps = NUM_INFERENCE_STEPS)

    # Generate the image
    image = pipe(
        prompt, 
        num_images_per_prompt = 1, 
        height = HEIGHT,
        width = WIDTH).images[0]
    
    # Calculate the time elapsed
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    # Save the image
    epoch_time = int(time.time()) # 10 characters (until the year 2286)
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:220]

    filename = "{}_{}".format(epoch_time, prompt)
    if not os.path.exists('output'):
        os.makedirs('output')

    # remove any / or . from file name
    filename = filename.replace("/", "_")
    filename = filename.replace(".", "_")
    filename = filename + ".png"

    image.save('output/' + filename)

    return filename, time_elapsed

if __name__ == '__main__':
    main()