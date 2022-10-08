import time
from happytransformer import HappyGeneration, GENSettings, GENTrainArgs
import os

# SECTION Journal
#
# - TODO insert a feedback system to save the positive results and 
#        call train() on them
# - TODO Create a corpus to start with some informations
# - TODO Find a way to keep the model loaded in memory
#
# !SECTION Journal

# INSTRUCTIONS:
# declare an instance using
# gpt = neo.GPTNeo([model=model_type_as_below])
# where model_type_as_below is one of the described below
# then you can either train the model on a dataset using
# gpt.train(dataset_path)
# or generate text using
# gpt.generate(prompt)
# you can also override the default settings using
# gpt.set_parameters([parameter=value])
# where parameter is one included in the list below
# and you can override training settings using
# gpt.train(dataset_path, [parameter=value])
# where parameter is one included in the list below

# TODO: Improving flexibility through loading and saving preprocessed data

# NOTE Advanced Settings Manual
'''
Parameter	Default	Definition

min_length	10	Minimum number of generated tokens

max_length	50	Maximum number of generated tokens

do_sample	False	When True, picks words based on their conditional probability

early_stopping	False	When True, generation finishes if the EOS token is reached

num_beams	1	Number of steps for each search path

temperature	1.0	How sensitive the algorithm is to selecting low probability options

top_k	50	How many potential answers are considered when performing sampling

top_p	1.0	Min number of tokens are selected where their probabilities add up to top_p

no_repeat_ngram_size	0	The size of an n-gram that cannot occur more than once. (0=infinity)

bad_words	None	List of words/phrases that cannot be generated.
'''

# NOTE Training arguments
'''
learning_rate: How much the model’s weights are adjusted per step. Too low and the model will take a long time to learn or get stuck in a suboptimal solution. Too high can cause can divergent behaviors.

num_train_epochs: The number of times the training data is iterated over.

weight_decay: A type of regularization. It prevents weights from getting too large. Thus, preventing overfitting.

adam_beta1: The beta1 parameter for the Adam with weight decay optimizer.

adam_beta2: The beta2 parameter for the Adam with weight decay optimizer.

adam_epsilon: The epsilon parameter for the Adam with weight decay optimizer.

max_grad_norm: Used to prevent exploding gradients. Prevents the derivatives of the loss function from exceed the absolute value of “max_grad_norm”.

batch_size: Number of training examples used per iteration

fp16: If true, enables half precision training which saves space by using 16 bits instead of 32 to store the model’s weights. Only available when CUDA/a a GPU is being used.
'''



class GPTNeo:

    def __init__(self, model="neo-small"):
        # ANCHOR Preparing the model
        self.model = None
        self.model_name = None
        self.model_folder = model
        self.set_model(model)
        # ANCHOR Checking if model exists and loading it
        final_folder = "models/" + self.model_folder
        die = False
        completed = False
        exception = None
        while (not die) and (not completed):
            try:
                if not os.path.exists(final_folder + "/pytorch_model.bin"):
                    self.gen = HappyGeneration(self.model_name,
                                            self.model)
                    self.gen.save(final_folder)
                    completed = True
                else:
                    self.gen = HappyGeneration(self.model_name,
                                            self.model,
                                            load_path=final_folder)
                    completed = True
            except Exception as e:
                if "Connection broken" in str(e):
                    print("[!] Connection broken! Retrying in 5 seconds...")
                    time.sleep(5)
                elif "timed out" in str(e):
                    print("[x] Timed out. Waiting 60 seconds....")
                    time.sleep(60)
                else:
                    exception = str(e)
                    die = True
        # Check if is to continue
        if die:
            raise Exception ("[!] " + exception)
        # ANCHOR Default settings
        self.settings = None
        self.setted = False

    # ANCHOR Model selection
    def set_model(self, model):
        # Supporting either small/medium/large and 125M/1.3B/2.7B
        if model == "neo-small":
            self.model = "EleutherAI/gpt-neo-125M"  # Approx 600MB
            self.model_name = "GPT-NEO"
        elif model == "neo-medium":
            self.model = "EleutherAI/gpt-neo-1.3B"  # Approx 5GB
            self.model_name = "GPT-NEO"
        elif model == "neo-large":
            self.model = "EleutherAI/gpt-neo-2.7B"  # Approx 10GB
            self.model_name = "GPT-NEO"
        elif model == "neox":
            self.model = "EleutherAI/gpt-neox-20b"  # Approx 45GB
            self.model_name = "GPT-NEOX"
        elif model == "dialo-small":
            self.model = "microsoft/DialoGPT-small"
            self.model_name = "DialoGPT-Small"
        elif model == "dialo-medium":
            self.model = "microsoft/DialoGPT-medium"
            self.model_name = "DialoGPT-Medium"
        elif model == "dialo-large":
            self.model = "microsoft/DialoGPT-large"
            self.model_name = "DialoGPT-Large"
        elif model == "rag":
            self.model = "facebook/rag-token-nq"
            self.model_name = "RAG"
        elif model == "blender-small":
            self.model = "facebook/blenderbot_small-90M"
            self.model_name = "BLENDER-SMALL"
        elif model == "blender-medium":
            self.model = "facebook/blenderbot-400M-distill"
            self.model_name = "BLENDER-MEDIUM"
        elif model == "blender-large":
            self.model = "facebook/blenderbot-1B-distill"
            self.model_name = "BLENDER-LARGE"
        elif model == "blender-huge":
            self.model = "facebook/blenderbot-3B"
            self.model_name = "BLENDER-HUGE"
        elif model == "aeona":
            self.model = "deepparag/Aeona"
            self.model_name = "AEONA"
        else:
            raise Exception("Invalid model")

    # ANCHOR Custom parameters support
    def set_parameters(self,
                       min_length=10,
                       max_length=50,
                       do_sample=True,
                       early_stopping=False,
                       num_beams=1,
                       temperature=0.7,
                       top_k=50,
                       top_p=1.0,
                       no_repeat_ngram_size=1,
                       bad_words=None
                       ):

        self.setted = True

        self.min_length = min_length
        self.max_length = max_length
        self.do_sample = do_sample
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.bad_words = bad_words

    # ANCHOR Training on a file with custom settings as above
    def train(self,
              file,
              load=False,
              epochs=1,
              # TODO Add other parameters
              ):
        # Training the model
        # TODO Add other parameters
        # NOTE Supporting plain loading of preprocessed data
        if not load:
            # NOTE Training and saving data for the next time
            train_settings = GENTrainArgs(num_train_epochs=epochs,
                                          save_preprocessed_data=True,
                                          save_preprocessed_data_path="models/" +
                                          self.model_folder +
                                          "/preprocessed_data/preprocess.json")
        else:
            if os.path.exists("models/" + self.model_folder +
                              "/preprocessed_data/preprocess.json"):
                # NOTE Loading a preprocessed file
                train_settings = GENTrainArgs(num_train_epochs=epochs,
                                              load_preprocessed_data=True,
                                              load_preprocessed_data_path="models/" +
                                              self.model_folder +
                                              "/preprocessed_data/preprocess.json")
            else:
                # NOTE Silently avoid loading if not existing
                return

        # NOTE Training process
        self.gen.train(file, args=train_settings)

    # ANCHOR Actual generation
    def generate(self, initial):

        # Setting the input to predict
        input = initial

        # Generating the reply
        if self.setted:
            # Initializing the settings
            self.settings = GENSettings(
                min_length=self.min_length,
                max_length=self.max_length,
                do_sample=self.do_sample,
                early_stopping=self.early_stopping,
                num_beams=self.num_beams,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                bad_words=self.bad_words
            )
            result = self.gen.generate_text(input,
                                            args=self.settings)
        else:
            self.settings = GENSettings(no_repeat_ngram_size=2,
                                        do_sample=True,
                                        top_k=50,
                                        temperature=0.7)
            result = self.gen.generate_text(input, args=self.settings)
            one_line_result = result.text.split('\n')[0]
            if one_line_result.strip() == "":
                one_line_result = result.text.split('\n')[1]
        return one_line_result, result
