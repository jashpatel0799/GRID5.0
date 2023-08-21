# import requirements after insatll it from requirements.txt
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
from transformers import pipeline
from PIL import Image 


# main model from huggingface pretrain model
## stable-diffusion-2-1 by stabilityai which generate image from text prompt
## previous used model which not proper work#"runwayml/stable-diffusion-v1-5"

# set model id
model_id = "stabilityai/stable-diffusion-2-1"

# take pretrain model from hugginface and pass through pipeline scheduler
_pipe = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

# Optimized the model
_pipe.scheduler = DPMSolverMultistepScheduler.from_config(_pipe.scheduler.config)
# _pipe = _pipe.to("cuda:0")


# here come second model in picture
# we can't find customer personalize dataset of any ecomerse of dummy at dataset website like kaggle, uci etc
# so we believe we get some keyword feature from customer data like age, color product looking for, gender, which type of cloths more purchase
# and current trend from google and social media


####### ----------------------------------------------------------------------------------------------------------------------------------
# you can play freelywith below variable
color = "red"
current_trend = "oversize"
gender = "male"
age = "24"
cloths = "shirt"
inf_step = 10 # is number of inference step our main model need to perform
# in our hyperparameter tuning 25 is best inference step number but you can play around with it loww number give poor result and high number give perfect result but it may take more time
# for 1 inference step it too around 6-7 mins on CPU and on GPU(Google CoLab) it takes around 1 min for 20 inference steps
# time may depend apone CPU and GPU (model)

####### ----------------------------------------------------------------------------------------------------------------------------------

# Now using that key word we create raw prompt which we call text_prompt
# which is used in our second pretained generative model which generate text to text and generate best prompt for our main stable diffusion model which generate fashion image
text_prompt = f"write a prompt for image generation of fashion recommendation for {age} years old {gender} for {cloths} in {color} with {current_trend} trends"


# second pretraind text to text model 
#flan-alpaca-gpt4-xl by declare-lab also take from hugging face
model = pipeline(model = "declare-lab/flan-alpaca-gpt4-xl")


# now pass raw prompt to model and get generated prompt from model
gen_prompt = model(text_prompt, max_length=64, do_sample=True)


# genrated prompt is list of dictionary from which we get generate prompt
prompt = gen_prompt[0]["generated_text"]
# print(prompt)


def get_inputs(batch_size=1):
    # generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = inf_step

    return {"prompt": prompts, "num_inference_steps": num_inference_steps}


# now we pass our generated prompt to iage generator model and show the result to the customer
# here we saved the result 
images = _pipe(**get_inputs(batch_size=10)).images # list of images


# here the finctionality to save rhe generated images and images are in type PIL
i = 1
for img in images:
    img.save(f"img_{i}.jpeg")
    i += 1
    
# make_image_grid(images, 2, 5)