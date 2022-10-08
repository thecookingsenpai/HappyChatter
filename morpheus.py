import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class GPTNeo:

    # NOTE initialize with model=None to avoid downloading
    #      you will need to download the model manually
    def __init__(self, model="neo-small"):
            self.model_map = {
                "neo-small": {
                    "string": "EleutherAI/gpt-neo-125M",
                    "name": "GPT-NEO"
                },
                "neo-medium": {
                    "string": "EleutherAI/gpt-neo-1.3B",
                    "name": "GPT-NEO"
                },
                "neo-large": {
                    "string": "EleutherAI/gpt-neo-2.7B",
                    "name": "GPT-NEO"
                },
                "neox": {
                    "string": "EleutherAI/gpt-neox-20b",
                    "name": "GPT-NEOX"
                },
                "dialo-small": {
                    "string": "microsoft/DialoGPT-small",
                    "name": "DialoGPT-Small"
                },
                "dialo-medium": {
                    "string": "microsoft/DialoGPT-medium",
                    "name": "DialoGPT-Medium"
                },
                "dialo-large": {
                    "string": "microsoft/DialoGPT-large",
                    "name": "DialoGPT-Large"
                },
                "rag": {
                    "string": "facebook/rag-token-nq",
                    "name": "RAG"
                },
                "blender-small": {
                    "string": "facebook/blenderbot_small-90M",
                    "name": "BLENDER-SMALL"
                },
                "blender-medium": {
                    "string": "facebook/blenderbot-400M-distill",
                    "name": "BLENDER-MEDIUM"
                },
                "blender-large": {
                    "string": "facebook/blenderbot-1B-distill",
                    "name": "BLENDER-LARGE"
                },
                "blender-huge": {
                    "string": "facebook/blenderbot-3B",
                    "name": "BLENDER-HUGE"
                },
                "aeona": {
                    "string": "deepparag/Aeona",
                    "name": "AEONA"
                }
            }
            # Can initialize without downloading
            if model:
                self.model_string = self.model_map.get(model).get("string")
                self.model_name = self.model_map.get(model).get("name")
                if not self.model_string:
                    raise Exception("Model not found.")
                self.tokenizer, self.model = self.load_tokenizer_and_model(self.model_string)

    # ANCHOR Manual download
    def download_model(self, model="neo-small"):
        self.model_string = self.model_map.get(model).get("string")
        self.model_name = self.model_map.get(model).get("name")
        if not self.model_string:
            raise Exception("Model not found.")
        self.tokenizer, self.model = self.load_tokenizer_and_model(self.model_string)

    # ANCHOR Download / Load model and tokenizer
    def load_tokenizer_and_model(self, model="microsoft/DialoGPT-small"):
        """
            Load tokenizer and model instance for some specific DialoGPT model.
        """
        # Initialize tokenizer and model
        print("Loading model...")
        tokenizer_folder = "saved/" + model + "/tokenizer"
        model_folder = "saved/" + model + "/model"
        try: 
            os.makedirs(tokenizer_folder)
        except:
            pass
        try:
            os.makedirs(model_folder)
        except:
            pass
        if os.path.exists(tokenizer_folder):
            try:
                tokenizer_inst = AutoTokenizer.from_pretrained(tokenizer_folder)
            except:
                os.removedirs(tokenizer_folder)
                os.makedirs(tokenizer_folder)
                tokenizer_inst = AutoTokenizer.from_pretrained(model)
                tokenizer_inst.save_pretrained("saved/" + model + "/tokenizer")   
        else:
            tokenizer_inst = AutoTokenizer.from_pretrained(model)
            tokenizer_inst.save_pretrained("saved/" + model + "/tokenizer")
        if os.path.exists(model_folder):
            try:
                model_inst = AutoModelForCausalLM.from_pretrained(model_folder)
            except:
                os.removedirs(model_folder)
                os.makedirs(model_folder)
                model_inst = AutoModelForCausalLM.from_pretrained(model)
                model_inst.save_pretrained("saved/" + model + "/model")
        else:
            model_inst = AutoModelForCausalLM.from_pretrained(model)
            model_inst.save_pretrained("saved/" + model + "/model")
        
        # Return tokenizer and model
        return tokenizer_inst, model_inst


        """
        # INFO Parameters to play with
        
        input_ids – (optional) torch.LongTensor of shape (batch_size, sequence_length) The sequence used as a prompt for the generation. If None the method initializes it as an empty torch.LongTensor of shape (1,).

        max_length – (optional) int The max length of the sequence to be generated. Between min_length and infinity. Default to 20.

        min_length – (optional) int The min length of the sequence to be generated. Between 0 and infinity. Default to 0.

        do_sample – (optional) bool If set to False greedy decoding is used. Otherwise sampling is used. Defaults to False as defined in configuration_utils.PretrainedConfig.

        early_stopping – (optional) bool if set to True beam search is stopped when at least num_beams sentences finished per batch. Defaults to False as defined in configuration_utils.PretrainedConfig.

        num_beams – (optional) int Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

        temperature – (optional) float The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

        top_k – (optional) int The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

        top_p – (optional) float The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

        repetition_penalty – (optional) float The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

        pad_token_id – (optional) int Padding token. Default to specicic model pad_token_id or None if it does not exist.

        bos_token_id – (optional) int BOS token. Defaults to bos_token_id as defined in the models config.

        eos_token_id – (optional) int EOS token. Defaults to eos_token_id as defined in the models config.

        length_penalty – (optional) float Exponential penalty to the length. Default to 1.

        no_repeat_ngram_size – (optional) int If set to int > 0, all ngrams of size no_repeat_ngram_size can only occur once.

        bad_words_ids – (optional) list of lists of int bad_words_ids contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use tokenizer.encode(bad_word, add_prefix_space=True).

        num_return_sequences – (optional) int The number of independently computed returned sequences for each element in the batch. Default to 1.

        attention_mask (optional) –

        torch.LongTensor of same shape as input_ids Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens. Defaults to None.

        What are attention masks?

        decoder_start_token_id=None – (optional) int If an encoder-decoder model starts decoding with a different token than BOS. Defaults to None and is changed to BOS later.

        use_cache – (optional) bool If use_cache is True, past key values are used to speed up decoding if applicable to model. Defaults to True.

        model_specific_kwargs – (optional) dict Additional model specific kwargs will be forwarded to the forward function of the model.
        """
        
    # ANCHOR Generate a response with parameters
    def generate(self, chat_round, chat_history_ids):
        """
            Generate a response to some user input.
        """
        # Encode user input and End-of-String (EOS) token
        new_input_ids = self.tokenizer.encode(input(">> You: ") + self.tokenizer.eos_token, return_tensors='pt')

        # Append tokens to chat history
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids
        
        # NOTE See above parameters
        # Generate response given maximum chat length history of 1250 tokens
        chat_history_ids = self.model.generate(bot_input_ids, max_length=1250, pad_token_id=self.tokenizer.eos_token_id)
        
        # Print response
        response_generated = "Bot: {}".format(self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
        
        chat_round += 1
        
        # Return the response, the chat history ids and the round
        return response_generated, chat_history_ids, chat_round

    # ANCHOR Training
    def train(self, 
              path,
              epochs=1, 
              batch_size=1, 
              learning_rate=1e-4, 
              train_data=None, 
              eval_data=None, 
              save_path=None):
        """
        Train the chatbot for some number of epochs.
        """
        # TODO https://huggingface.co/docs/transformers/v4.22.1/en/training#train
        pass
    
    
    # ANCHOR Chat for n rounds (to test)
    def chat_for_n_rounds(self, n=5):
        """
        Chat with chatbot for n rounds (n = 5 by default)
        """
        
        # Initialize history variable
        chat_history_ids = None
        
        
        # Chat for n rounds
        current_round = 1
        for chat_round in range(n):
            print(chat_history_ids)
            response, chat_history_ids, current_round = self.generate(chat_round, chat_history_ids)
            print("Round " + str(current_round) + "| " + response)
            
if __name__ == '__main__':
    # Initialize chatbot
    neo = GPTNeo()
    neo.chat_for_n_rounds(5)