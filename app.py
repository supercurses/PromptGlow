import io
import asyncio
import time

from nicegui import run, ui
from agent_prompt import PromptAgent
from agent_flux import FluxAgent
from agent_review import ReviewAgent
from agent_sdxl import SDXLAgent
from tokenizer import Tokenizer


async def update_timer(label, start_time):
    while True:
        elapsed_time = time.time() - start_time
        label.set_text(f"Elapsed time: {elapsed_time:.2f} seconds")
        await asyncio.sleep(0.1)  # Update every 100ms

async def generate_prompts():
    """ uses prompt agent to generate an embellished prompt from the user's initial prompt """
    user_prompt = user_input.value
    user_style = styles[style_toggle.value]
    # Start up the spinner
    with prompt_textarea:
        spinner = ui.spinner('dots', size='xl')
        spinner.visible = True

    generated_prompt = await run.io_bound(prompt_agent.generate_prompt,
                                          prompt=user_prompt, style=user_style)
    prompt_textarea.value = generated_prompt
    spinner.visible = False


async def shrink_prompt():
    """ Asks the LLM to reduce the words in the prompt """
    prompt = prompt_textarea.value
    shrunken_prompt = await run.io_bound(prompt_agent.shrink_prompt,
                                         prompt=prompt)
    prompt_textarea.value = shrunken_prompt


async def improve_prompt():
    review_dialog.close()
    user_style = styles[style_toggle.value]
    improvements_prompt = ('Use the following improvements to make improvements to the following prompt:\n'
                           'Improvements: {} \n Prompt:{}'.format(review.content, prompt_textarea.value))
    with prompt_textarea:
        spinner = ui.spinner('dots', size='xl')
        spinner.visible = True
    generated_prompt = await run.io_bound(prompt_agent.generate_prompt,
                                          prompt=improvements_prompt, style=user_style)  # Actual prompt generation
    prompt_textarea.value = generated_prompt
    spinner.visible = False

def update_sequence_length():
    token_count = tokenizer.get_sequence_length(prompt_textarea.value)
    sequence_length.set_text("Sequence Length: {}/256".format(token_count))

async def generate_image():
    prompt = prompt_textarea.value
    with image:
        spinner = ui.spinner(size='xl')
        spinner.visible = True

    # Start the processing timer
    start_time = time.time()
    timer_task = asyncio.create_task((update_timer(stopwatch_label,start_time)))
    urls = await run.io_bound(flux_agent.generate_image, model="black-forest-labs/flux-schnell",
                              prompt=prompt, steps=4, controlnet=False, image_url=None)
    timer_task.cancel()
    # End the timer
    end_time = time.time()
    url = urls[0]
    image.source = url
    image_label.set_text(url)
    review_button.style('visibility: visible')
    sdxl_button.style('visibility: visible')
    flux_button.style('visibility: visible')
    spinner.visible = False
    stopwatch_label.set_text(f"Final time: {time.time() - start_time:.2f} seconds")
async def generate_image_flux_dev():
    prompt = prompt_textarea.value
    control_url = image.source
    with image:
        spinner = ui.spinner(size='xl')
        spinner.visible = True
    urls = await run.io_bound(flux_agent.generate_image,
                              model="xlabs-ai/flux-dev-controlnet"
                                    ":f2c31c31d81278a91b2447a304dae654c64a5d5a70340fba811bb1cbd41019a2",
                              prompt=prompt, steps=28,
                              controlnet=True, image_url=control_url)
    url = urls[0]
    image.source = url
    image_label.set_text(url)
    review_button.style('visibility: visible')
    sdxl_button.style('visibility: visible')
    spinner.visible = False

async def review_image():
    review_dialog.open()
    with review_dialog:
        review_spinner = ui.spinner(size='xl')
    review.content = await run.io_bound(review_agent.review_image, url=image_label.text)
    review_spinner.visible = False





async def generate_sdxl():
    # Start the processing timer
    start_time = time.time()
    timer_task = asyncio.create_task((update_timer(stopwatch_label, start_time)))
    await run.io_bound(sdxl_agent.img2img, img2img_prompt=prompt_textarea.value,
                       counter=1,
                       image_path="nicegui_img2img.png",
                       image_url=image_label.text,
                       adetailer=True
                       )
    timer_task.cancel()

prompts = []
flux_agent = FluxAgent()
prompt_agent = PromptAgent(local=False)
review_agent = ReviewAgent()
sdxl_agent = SDXLAgent()
tokenizer = Tokenizer()
styles = {1: 'photograph', 2: 'illustration', 3: 'oil painting'}

with ui.row().style('gap:10em').classes('w-full no-wrap'):
    with ui.column().classes('w-1/2 pl-20 pt-10'):
        # Create the input field and button
        user_input = ui.textarea('Enter your prompt:').props('autogrow').style(
            'width:75%')  # Store the input field in a variable for later access
        with ui.row():
            style_chip_1 = ui.chip('Photograph', icon='photo_camera', selectable=True, color='orange')
            style_chip_2 = ui.chip('Illustration', icon='gesture', selectable=True, color='blue')
            style_chip_3 = ui.chip('Oil Painting', icon='brush', selectable=True, color='green')
        ui.label(style_chip_1.text)
        style_toggle = ui.toggle(styles)
        ui.button('get a prompt', on_click=generate_prompts)
        prompt_textarea = ui.textarea(value="Prompt",
                                      on_change=lambda e: update_sequence_length()).props('autogrow').style('width:75%')

        # Create a placeholder label to display the generated prompt
        prompt_display = ui.label('Your generated prompt will appear here.')
        sequence_length = ui.label('Sequence Length')
        with ui.row():
            ui.button('shrink prompt', on_click=shrink_prompt)
            ui.button('generate image', on_click=generate_image)

    with ui.column().classes('w-1/2 pr-20 pt-10'):
        stopwatch_label = ui.label()
        image = ui.image().style('width: 100%; height: 60%; background-color: silver')
        image.source = 'placeholder.png'
        image_label = ui.label().style('visibility: hidden')
        with ui.row(align_items='center'):
            review_button = ui.button('rate image', on_click=review_image).style('visibility: hidden')
            sdxl_button = ui.button('send to sdxl', on_click=generate_sdxl).style('visibility: hidden')
            flux_button = ui.button('send to flux dev', on_click=generate_image_flux_dev).style('visibility: hidden')


with ui.dialog() as review_dialog, ui.card().style('width:50%; max-width: none'):
    review = ui.markdown()
    with ui.row():
        ui.button('Improve Prompt', on_click=improve_prompt)
        ui.button('Close', on_click=review_dialog.close)


ui.run()
