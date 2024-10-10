import asyncio
import time

from nicegui import run, ui, app
from agent_prompt import PromptAgent
from agent_flux import FluxAgent
from agent_review import ReviewAgent
from agent_sdxl import SDXLAgent
from tokenizer import Tokenizer

app.add_static_files('/images', 'images')


async def update_timer(label, start_time):
    """ Starts a timer and updates the given label with elapsed time"""
    while True:
        elapsed_time = time.time() - start_time
        label.set_text(f"Elapsed time: {elapsed_time:.2f} seconds")
        await asyncio.sleep(0.1)  # Update every 100ms


async def generate_prompts():
    """ uses prompt agent to generate an embellished prompt from the user's initial prompt """
    generated_prompt = None
    # Start up the spinner
    with prompt_textarea:
        spinner = ui.spinner('dots', size='xl')
        spinner.visible = True
    try:
        generated_prompt = await run.io_bound(prompt_agent.generate_prompt,
                                              art_type=art_type.value,
                                              media=media.value,
                                              prompt=user_prompt.value)
    except Exception as e:
        ui.notify(f'Unable to get a prompt: {e}', type='negative')
        print('timeout')
    finally:
        prompt_textarea.value = generated_prompt
        spinner.visible = False


async def shrink_prompt():
    """ Asks the LLM to reduce the words in the prompt """
    prompt = prompt_textarea.value
    shrunken_prompt = await run.io_bound(prompt_agent.shrink_prompt,
                                         prompt=prompt)
    prompt_textarea.value = shrunken_prompt


async def generate_clip_prompt():
    t5_prompt = prompt_textarea.value
    with sdxl_clip_prompt:
        spinner = ui.spinner('dots', size='xl')
        spinner.visible = True
    clip_prompt = await run.io_bound(prompt_agent.generate_clip_prompt, prompt=t5_prompt)
    sdxl_clip_prompt.value = clip_prompt
    spinner.visible = False


async def improve_prompt():
    """ Asks an LLM to improve the prompt based on the suggested improvements from the review """
    review_dialog.close()
    improvements_prompt = ('Use the following improvements to make improvements to the following prompt:\n'
                           'Improvements: {} \n Prompt:{}'.format(review.content, prompt_textarea.value))
    with prompt_textarea:
        spinner = ui.spinner('dots', size='xl')
        spinner.visible = True
    generated_prompt = await run.io_bound(prompt_agent.generate_prompt,
                                          art_type=art_type.value,
                                          media=media.value,
                                          prompt=improvements_prompt)  # Actual prompt generation
    prompt_textarea.value = generated_prompt
    spinner.visible = False


def update_sequence_length():
    """ Keeps the sequence length variable updated if the prompt is changed """
    if prompt_textarea.value:
        token_count = tokenizer.get_sequence_length(prompt_textarea.value)
        sequence_length.set_text("Sequence Length: {}/256".format(token_count))


async def generate_image():
    """ Uses Flux Agent to create an image from the prompt """
    prompt = prompt_textarea.value
    flux_generate_button.disable()
    timer_task = None
    urls = None
    start_time = time.time()
    with carousel_placeholder:
        spinner = ui.spinner(size='xl')
        spinner.visible = True
    try:
        # Start the processing timer

        timer_task = asyncio.create_task((update_timer(stopwatch_label, start_time)))
        urls = await run.io_bound(flux_agent.generate_image, model="black-forest-labs/flux-schnell",
                                  prompt=prompt, steps=4, controlnet=False, image_url=None)
    except Exception as e:
        ui.notify(f'Unable to get an image: {str(e)}', type='negative')
    finally:
        if timer_task:
            timer_task.cancel()
        flux_generate_button.enable()
        # End the timer
        end_time = time.time()
        if urls:
            url = urls[0]
            flux_image_urls.append(url)
        else:
            ui.notify('No valid image url returned', type='negative')
        review_button.style('visibility: visible')
        sdxl_button.style('visibility: visible')
        flux_button.style('visibility: visible')
        spinner.visible = False
        stopwatch_label.set_text(f"Final time: {time.time() - start_time:.2f} seconds")
        update_carousel()


async def generate_image_flux_dev():
    """ Creates a flux dev image from the prompt and uses the image as a controlnet """
    prompt = prompt_textarea.value
    control_url = flux_image_label.text
    timer_task = None
    start_time = time.time()
    urls = None
    with carousel_placeholder:
        spinner = ui.spinner(size='xl')
        spinner.visible = True
    try:
        timer_task = asyncio.create_task((update_timer(stopwatch_label, start_time)))
        urls = await run.io_bound(flux_agent.generate_image,
                                  model="xlabs-ai/flux-dev-controlnet"
                                        ":f2c31c31d81278a91b2447a304dae654c64a5d5a70340fba811bb1cbd41019a2",
                                  prompt=prompt, steps=28,
                                  controlnet=True, image_url=control_url)
    except Exception as e:
        ui.notify(f'Unable to get an image: {str(e)}', type='negative')
    finally:
        if timer_task:
            timer_task.cancel()
        # End the timer
        end_time = time.time()
        if urls:
            url = urls[0]
            flux_image_urls.append(url)
        review_button.style('visibility: visible')
        sdxl_button.style('visibility: visible')
        spinner.visible = False
        update_carousel()


async def review_image():
    """ Ask the LLM to provide suggestions on how to improve the image """
    review_dialog.open()
    with review_row:
        review_spinner = ui.spinner('dots', size='xl')
    try:
        review.content = await run.io_bound(review_agent.review_image, url=flux_image_label.text)
    except Exception as e:
        ui.notify(f'Unable to generate a review: {str(e)}', type='negative')
    finally:
        review_spinner.visible = False


def sdxl_dialog_manager():
    sdxl_dialog.open()


def update_carousel():
    carousel_placeholder.clear()
    with carousel_placeholder:
        for url in reversed(flux_image_urls):
            with ui.carousel_slide():
                ui.image(url).classes('w-[600px]')
    set_current_image()

def update_media():
    selected_art_type = art_type.value
    if selected_art_type == 'Animation':
        media.options = animation_media
    elif selected_art_type == 'Photograph':
        media.options = photograph_media
    elif selected_art_type == 'Drawing':
        media.options = drawing_media
    elif selected_art_type == 'Painting':
        media.options = painting_media
    elif selected_art_type == 'Pixel Art':
        media.options = other_media
    media.update()

def set_current_image():
    if not flux_image_urls:
        # Let's get out of here if we haven't created the list yet
        return
    slide_value = carousel_placeholder.value

    # Extract the index from the slide value (e.g., "slide_1" -> 1)
    slide_index = int(slide_value.replace('slide_', '')) - 1  # Subtract 1 to get zero-based index

    # Since the list was reversed when building the carousel, adjust the index accordingly
    original_index = len(flux_image_urls) - 1 - slide_index

    # Get the corresponding URL from the original list
    current_url = flux_image_urls[original_index]
    flux_image_label.set_text(current_url)
    sdxl_image.set_source(current_url)


async def generate_sdxl():
    """ Uses a local Automatic 1111 installation to create an SDXL image to image rendering """
    # Start the processing timer
    start_time = time.time()
    timer_task = asyncio.create_task((update_timer(sdxl_stopwatch_label, start_time)))
    file_path = None
    try:
        file_path = await run.io_bound(sdxl_agent.img2img, img2img_prompt=prompt_textarea.value,
                                       counter=1,
                                       image_path="nicegui_img2img.png",
                                       image_url=flux_image_label.text,
                                       adetailer=True
                                       )
    except Exception as e:
        ui.notify(f'Unable to get an image: {str(e)}', type='negative')
    finally:
        timer_task.cancel()
        if file_path:
            sdxl_image.source = file_path
        else:
            ui.notify('No valid file path returned to display', type='negative')


prompts = []
flux_agent = FluxAgent()
prompt_agent = PromptAgent(local=True)
review_agent = ReviewAgent()
sdxl_agent = SDXLAgent()
tokenizer = Tokenizer()

flux_image_urls = []
animation_media = ['Cut-Out', 'Claymation', 'Cel', 'Computer', 'Stop Motion', '3D Pixar', '3D']
photograph_media = ['Film', 'Digital']
drawing_media = ['Brush', 'Finger', 'Pen', 'Ballpoint Pen', 'Eraser', 'Fountain Pen', 'Technical Pen', 'Marker',
                 'Pencil', 'Colored Pencil', 'Charcoal', 'Crayon', 'Pastel', 'Conte', 'Chalk']
painting_media = ['Oil', 'Acrylic', 'Watercolor', 'Gouache', 'Coffee']
other_media = ['1Bit', '8bit', '16Bit']

with ui.row().style('gap:2em').classes('w-full no-wrap'):
    with ui.column().classes('w-1/2 pl-20 pt-10'):
        # Create the input field and button
        art_type = ui.select(['Animation', 'Photograph', 'Drawing', 'Painting','Pixel Art'], label='Art Type').style('width:75%').on_value_change(update_media)
        media = ui.select([], label='Media').style('width:75%')
        user_prompt = ui.textarea('Enter your prompt:').style(
            'width:75%')  # Store the input field in a variable for later access
        ui.button('get a prompt', on_click=generate_prompts)
        prompt_textarea = ui.textarea('Embellished Prompt',
                                      on_change=lambda e: update_sequence_length()).props('autogrow').style(
            'width:75%;')

        sequence_length = ui.label('Sequence Length')
        with ui.row():
            ui.button('shrink prompt', on_click=shrink_prompt)
            flux_generate_button = ui.button('generate image', on_click=generate_image)

    with ui.column().classes('w-1/2 pt-10'):
        stopwatch_label = ui.label()
        with ui.carousel(animated=True, arrows=True, navigation=True, on_value_change=set_current_image).props(
                "height=600px") as carousel_placeholder:
            with ui.carousel_slide():
                placeholder_image = ui.image('images/placeholder.png').classes('w-[600px] h-[600px]')

        with ui.row(align_items='center'):
            review_button = ui.button('rate image', on_click=review_image).style('visibility: hidden')
            sdxl_button = ui.button('SDXL', on_click=sdxl_dialog_manager).style('visibility: hidden')
            flux_button = ui.button('send to flux dev', on_click=generate_image_flux_dev).style('visibility: hidden')
            flux_image_label = ui.label().style('visibility: hidden')
with ui.dialog() as review_dialog, ui.card().style('width:50%; max-width: none') as review_card:
    with ui.row() as review_row:
        review = ui.markdown('Gathering feedback...')
    with ui.row():
        ui.button('Improve Prompt', on_click=improve_prompt)
        ui.button('Close', on_click=review_dialog.close)

with ui.dialog() as sdxl_dialog, ui.card().style('width:50%; max-width:none'):
    sdxl_stopwatch_label = ui.label('Elapsed time: ')
    sdxl_clip_prompt = ui.textarea().props('autogrow').style('width: 100%')
    ui.button('Get CLIP', on_click=generate_clip_prompt).style('visibility: visible')
    sdxl_image = ui.image().style('height: 60%')
    sdxl_go_button = ui.button('go', on_click=generate_sdxl).style('visibility: visible')

ui.run()
