import PySimpleGUI as sg
import os
import neo
import uuid
import base64

# NOTE Model description helper


def get_model_description(model):
    descriptions = {
        "SMALL": '''
            The smallest model by EleuterAI, based on a GPT-3 like
            architecture. Is the fastest but less accurate model and
            it weights around 500MB. It contains 125M pre trained
            parameters.
        ''',
        "MEDIUM": '''
            The mid-sized model by EleuterAI, based on a GPT-3 like
            architecture. It is balanced in both speed and accuracy.
            It weights around 5GB and contains 1.3B pre trained
            parameters.
        ''',
        "LARGE": '''
            A large and very accurate model by EleuterAI. 
            Based on GPT-3 like architecture, it is trained
            to be high performant and contains around 2.7B
            pre trained parameters. It weights around 11GB.
        ''',
        "NEOX": '''
            The newest model by EleuterAI, and the largest one
            so far. Based on GPT-3 like architecture it contains
            20B pre trained parameters for a total of 45GB size.
            It is by far the most accurate one in this family.
        ''',
        "DIALO-SMALL": '''
            The smallest and fastest model by Microsoft, built
            to be used specifically in conversations. Accuracy
            varies a lot and is good to be trained some more.
            It is based on GPT-2.
        ''',
        "DIALO-MEDIUM": '''
            Mid sized and mid accuracy model by Microsoft based
            on GPT-2. Is a good compromise between performances
            and accuracy in most cases.
        ''',
        "DIALO-LARGE": '''
            The largest model by Microsoft based on GPT-2.
            It is the most precise of its family and of course
            the most heavy one to store and use.
        ''',
        "RAG": '''
            An experimental model by Facebook. Results vary a lot
            and is not yet well studied. Can be used to experiment
            or to train onto.
        ''',
        "BLENDER-SMALL": '''
            The tiny member of the Blenderbot family, a Facebook
            creation that aims to be the best conversational model.
        ''',
        "BLENDER-MEDIUM": '''
            The mid-sized member of the Blenderbot family, a Facebook
            creation that aims to be the best conversational model.
        ''',
        "BLENDER-LARGE": '''
            The large member of the Blenderbot family, a Facebook
            creation that aims to be the best conversational model.
        ''',
        "BLENDER-HUGE": '''
            The most complete member of the Blenderbot family, a Facebook
            creation that aims to be the best conversational model.
        ''',
        "AEONA": '''
            A DialoGPT based model that improves the original one in many
            fields and is aimed to be the most natural possible.
        '''
    }
    return descriptions.get(model)


# NOTE Changing to current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())

# NOTE: Creating the structure needed if is not present
if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("logs"):
    os.mkdir("logs")
if not os.path.exists("models/dialo-small"):
    os.makedirs("models/dialo-small/preprocessed_data")
if not os.path.exists("models/dialo-medium"):
    os.makedirs("models/dialo-medium/preprocessed_data")
if not os.path.exists("models/dialo-large"):
    os.makedirs("models/dialo-large/preprocessed_data")
if not os.path.exists("models/neo-small"):
    os.makedirs("models/neo-small/preprocessed_data")
if not os.path.exists("models/neo-medium"):
    os.makedirs("models/neo-medium/preprocessed_data")
if not os.path.exists("models/neo-large"):
    os.makedirs("models/neo-large/preprocessed_data")
if not os.path.exists("models/neox"):
    os.makedirs("models/neox/preprocessed_data")
if not os.path.exists("models/rag"):
    os.makedirs("models/rag/preprocessed_data")
if not os.path.exists("models/blender-small"):
    os.makedirs("models/blender-small/preprocessed_data")
if not os.path.exists("models/blender-medium"):
    os.makedirs("models/blender-medium/preprocessed_data")
if not os.path.exists("models/blender-large"):
    os.makedirs("models/blender-large/preprocessed_data")
if not os.path.exists("models/blender-huge"):
    os.makedirs("models/blender-huge/preprocessed_data")
if not os.path.exists("models/aeona"):
    os.makedirs("models/aeona/preprocessed_data")

# ANCHOR Save chat method


def save_chat(logfile, In, Out):
    with open(logfile, "a") as logfile_stream:
        logfile_stream.write(In + "\n" + Out + "\n")


# ANCHOR Entry point
if __name__ == "__main__":
    # NOTE Generating a random logfile name
    folder = "logs"
    rname = str(uuid.uuid4())
    filename = folder + "/" + rname
    # NOTE Preparing GUI parameters
    MLINE_KEY = '-ML-'+sg.WRITE_ONLY_KEY
    # NOTE Setting the default model description
    model_description = get_model_description("SMALL")
    output = "Output:\n\n"
    # NOTE You can set an initial imprinting here
    log = '''This is a conversation between a smart and curious Bot
             and a Human. The bot is self conscious and creative.
             The Bot is original and doesn't repeat the same sentences.\n\n
             Human: Hello!\n
             Bot: Hello there!\n
             Human: How are you?\n
             Bot: Fine! And you?\n
             Human: Who are you?\n
             Bot: I am a sentient bot, and you?\n'''
    # NOTE Creating the logfile included initial logs
    with open(filename, "w+") as logfile_stream:
        logfile_stream.write("HappyChatter Logfile\n\n\n" + log)
    # NOTE Building the GUI
    sg.theme('Dark2')
    font = ("Arial", 12)
    sg.set_options(font=font)
    icon = os.getcwd() + "/data/icon.png"
    sg.set_options(icon=base64.b64encode(open(icon, 'rb').read()))
    layout = [[sg.Text('HappyChatter AI Powered Chatbot')],
              [[sg.Multiline(output, size=(80, 20), key=MLINE_KEY)],
               [[sg.Text('Write something')],
                [sg.Input(key='-IN-')]],
               [[sg.Text('Status: Ready', key="Status")],
               [sg.Button('Send'), sg.Exit()]]],
              [sg.HorizontalSeparator()],
              [sg.Text("Train on a file: "), sg.FileBrowse(key="-TRAINFILE-")],
              [sg.Button('Train')],
              [sg.HorizontalSeparator()],
              [sg.Text("Model Selection")],
              [sg.HorizontalSeparator()],
              [sg.Text("EleuterAI Models")],
              [sg.Radio('GPT-NEO Small', "MODEL", enable_events=True, default=True, key="SMALL"),
               sg.Radio('GPT-NEO Medium', "MODEL", enable_events=True,
                        default=False, key="MEDIUM"),
               sg.Radio('GPT-NEO Large', "MODEL", enable_events=True,
                        default=False, key="LARGE"),
               sg.Radio('NEOX', "MODEL", enable_events=True, default=False, key="NEOX")],
              [sg.HorizontalSeparator()],
              [sg.Text("Microsoft Models")],
              [sg.Radio('DialoGPT Small', "MODEL", enable_events=True, default=False, key="DIALO-SMALL"),
               sg.Radio('DialoGPT Medium', "MODEL", enable_events=True,
                        default=False, key="DIALO-MEDIUM"),
               sg.Radio('DialoGPT Large', "MODEL", enable_events=True, default=False, key="DIALO-LARGE")],
              [sg.HorizontalSeparator()],
              [sg.Text("Facebook models")],
              [sg.Radio('RAG', "MODEL", enable_events=True,
                        default=False, key="RAG")],
              [sg.Radio('Blenderbot Small', "MODEL", enable_events=True, default=False, key="BLENDER-SMALL"),
               sg.Radio('Blenderbot Medium', "MODEL", enable_events=True,
                        default=False, key="BLENDER-MEDIUM"),
               sg.Radio('Blenderbot Large', "MODEL", enable_events=True,
                        default=False, key="BLENDER-LARGE"),
               sg.Radio('Blenderbot Huge', "MODEL", enable_events=True, default=False, key="BLENDER-HUGE")],
              [sg.HorizontalSeparator()],
              [sg.Text("Miscellaneous")],
              [sg.Radio('Aeona (by deepparag)', "MODEL",
                        enable_events=True, default=False, key="AEONA")],
              [sg.HorizontalSeparator()],
              [sg.Text(model_description, key="Description")],
              [sg.HorizontalSeparator()],
              [sg.Text("Coded by TheCookingSenpai")],
              [sg.Button("Credits", key="CREDITS")]]

    print("ICON: " + icon)
    window = sg.Window('HappyChatter', layout=layout)

    # ANCHOR Event loop
    while True:
        # NOTE Reading events from the GUI
        event, values = window.read()
        print(event, values)
        # NOTE Managing close window
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        # NOTE Text input event
        elif event == "Send":
            # NOTE Getting the input
            text_input = values["-IN-"]
            # NOTE Updating the output and the chat log
            log += "Human: " + text_input + "\n"
            output += "Human: " + text_input + "\n"
            window[MLINE_KEY].update(output)
            window["Status"].update(
                "Status: loading model, please be patient...")
            window.refresh()
            # NOTE Model preparation
            model_chosen = None
            if values["SMALL"]:
                model_chosen = "neo-small"
            elif values["MEDIUM"]:
                model_chosen = "neo-medium"
            elif values["LARGE"]:
                model_chosen = "neo-large"
            elif values["NEOX"]:
                model_chosen = "neox"
            elif values["DIALO-SMALL"]:
                model_chosen = "dialo-small"
            elif values["DIALO-MEDIUM"]:
                model_chosen = "dialo-medium"
            elif values["DIALO-LARGE"]:
                model_chosen = "dialo-large"
            elif values["RAG"]:
                model_chosen = "rag"
            elif values["BLENDER-SMALL"]:
                model_chosen = "blender-small"
            elif values["BLENDER-MEDIUM"]:
                model_chosen = "blender-medium"
            elif values["BLENDER-LARGE"]:
                model_chosen = "blender-large"
            elif values["BLENDER-HUGE"]:
                model_chosen = "blender-huge"
            elif values["AEONA"]:
                model_chosen = "aeona"
            # NOTE Model loading
            print("[*] Loading model...")
            gpt = neo.GPTNeo(model=model_chosen)
            print("[+] Model loaded.")
            window["Status"].update(
                "Status: model loaded. Generating response...")
            window.refresh()
            # NOTE Loading existant training data if any
            gpt.train("", load=True)
            # NOTE Chatbot response
            # NOTE Sending the whole log to try to keep coherency
            result, raw = gpt.generate(log)
            print("RAW DATA:")
            print(raw)
            print("RAW RESULT: " + result)
            result = result.strip()
            log += "Bot: " + result + "\n"
            output += "Bot: " + result + "\n"
            window[MLINE_KEY].update(output)
            window["Status"].update("Status: saving the response")
            window.refresh()
            result = "Bot: " + result
            # NOTE Saving the logs
            save_chat(filename, "Human: " + text_input, result)
            window["Status"].update("Status: ready")
            window.refresh()
            print("[+] Done. Ready for another round")
        # NOTE Train event
        elif event == "Train":
            if values["-TRAINFILE-"] == "":
                pass
            else:
                window["Status"].update("Status: sending training...")
                window.refresh()
                gpt.train(values["-TRAINFILE-"])
                window["Status"].update("Status: ready")
                window.refresh()
        # NOTE Radio button change events
        elif ((event == "SMALL") or
              (event == "MEDIUM") or
              (event == "LARGE") or
              (event == "BLENDER-SMALL") or
              (event == "BLENDER-MEDIUM") or
              (event == "BLENDER-LARGE") or
              (event == "BLENDER-HUGE") or
              (event == "DIALO-SMALL") or
              (event == "DIALO-MEDIUM") or
              (event == "DIALO-LARGE") or
              (event == "RAG") or
              (event == "AEONA")):
            model_description = get_model_description(event)
            window["Description"].update(model_description)
            window.refresh()
        elif event == "Credits":
            window.Hide()
            sg.Popup('''
                     All the models are to be credited to their
                     authors as reported in their descriptions.
                     The models are downloaded from https://huggingface.co/
                     This software uses PySimpleGUI and Happytransform
                     libraries to provide a nice user experience and the
                     availability of the latest models distributed.
                     
                     This software has been coded by TheCookingSenpai
                     (http://github.com/thecookingsenpai) and is 
                     distributed under Creative Commons CC-BY-ND license.
                     ''')
            window.Show()
    # NOTE Event close
    window.close()
