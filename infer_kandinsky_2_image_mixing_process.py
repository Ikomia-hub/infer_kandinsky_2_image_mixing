import copy
from ikomia import core, dataprocess
import torch
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
from PIL import Image
import numpy as np
import random
import os


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferKandinsky2ImageMixingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.prompt = "a cat"
        self.prior_guidance_scale = 4.0
        self.guidance_scale = 1.0
        self.negative_prompt = ""
        self.height = 768
        self.width = 768
        self.prior_num_inference_steps = 25
        self.weights = "[0.3, 0.3, 0.4]"
        self.num_inference_steps = 100
        self.seed = -1
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.prompt = str(param_map["prompt"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.prior_guidance_scale = float(param_map["prior_guidance_scale"])
        self.negative_prompt = str(param_map["negative_prompt"])
        self.seed = int(param_map["seed"])
        self.height = int(param_map["height"])
        self.width = int(param_map["width"])
        self.weights = str(param_map["weights"])
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.prior_num_inference_steps = int(param_map["prior_num_inference_steps"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["prompt"] = str(self.prompt)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["prior_guidance_scale"] = str(self.prior_guidance_scale)
        param_map["height"] = str(self.height)
        param_map["width"] = str(self.width)
        param_map["weights"] = str(self.weights)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["prior_num_inference_steps"] = str(self.prior_num_inference_steps)
        param_map["seed"] = str(self.seed)

        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferKandinsky2ImageMixing(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_input(dataprocess.CImageIO())
        self.add_input(dataprocess.CImageIO())
        self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferKandinsky2ImageMixingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.pipe = None
        self.pipe_prior = None
        self.generator = None
        self.seed = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
        self.model_name = "kandinsky-community/kandinsky-2-2-decoder"
        self.prior_model_name = "kandinsky-community/kandinsky-2-2-prior"


    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def generate_seed(self, seed):
        if seed == -1:
            self.seed = random.randint(0, 191965535)
        else:
            self.seed = seed
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

    def load_pipeline(self, local_files_only=False, prior=False):
        torch_tensor_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if prior:
            pipeline = KandinskyV22PriorPipeline.from_pretrained(
                self.prior_model_name,
                torch_dtype=torch_tensor_dtype,
                cache_dir=self.model_folder,
                local_files_only=local_files_only
                )
        else:
            pipeline = KandinskyV22Pipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch_tensor_dtype,
                cache_dir=self.model_folder,
                local_files_only=local_files_only
                )

        pipeline.to(self.device)

        return pipeline

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Get input
        input_img1 = self.get_input(0).get_image()
        input_img2 = self.get_input(1).get_image()

        if param.update or self.pipe_prior is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.generate_seed(param.seed)

            try:
                self.pipe_prior = self.load_pipeline(local_files_only=True, prior=True)
            except Exception as e:
                self.pipe_prior = self.load_pipeline(local_files_only=False, prior=True)

            param.update = False

        # Convert image
        img1 = Image.fromarray(input_img1)
        img2 = Image.fromarray(input_img2)

        # add all the conditions we want to interpolate, can be either text or image
        images_texts = [param.prompt, img1, img2]

        # specify the weights for each condition in images_texts
        weights = eval(param.weights)

        with torch.no_grad():
            prior_out = self.pipe_prior.interpolate(
                                        images_and_prompts=images_texts,
                                        weights=weights,
                                        num_inference_steps=param.prior_num_inference_steps,
                                        guidance_scale=param.prior_guidance_scale,
                                        negative_prompt=param.negative_prompt,
                                        generator=self.generator)

        if self.pipe is None:
            try:
                self.pipe = self.load_pipeline(local_files_only=True, prior=False)
            except Exception as e:
                self.pipe = self.load_pipeline(local_files_only=False, prior=False)

        with torch.no_grad():
            result = self.pipe(
                            **prior_out,
                            num_inference_steps=param.num_inference_steps,
                            guidance_scale=param.guidance_scale,
                            generator=self.generator,
                            height=param.height,
                            width=param.width,
                ).images[0]

        # Get and display output
        image = np.array(result)
        output_img = self.get_output(0)
        output_img.set_image(image)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferKandinsky2ImageMixingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_kandinsky_2_image_mixing"
        self.info.short_description = "Kandinsky 2.2 image mixing diffusion model."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/einstein.jpg"
        self.info.authors = "A. Shakhmatov, A. Razzhigaev, A. Nikolich, V. Arkhipkin, I. Pavlov, A. Kuznetsov, D. Dimitrov"
        self.info.article = "https://aclanthology.org/2023.emnlp-demo.25/"
        self.info.journal = "ACL Anthology"
        self.info.year = 2023
        self.info.license = "Apache 2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"
        # Code source repository
        self.info.repository = "https://github.com/ai-forever/Kandinsky-2"
        # Keywords used for search
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "IMAGE_GENERATION"
        self.info.keywords = "Latent Diffusion,Hugging Face,Kandinsky,Image mixing,Interpolation,Generative"

    def create(self, param=None):
        # Create algorithm object
        return InferKandinsky2ImageMixing(self.info.name, param)
