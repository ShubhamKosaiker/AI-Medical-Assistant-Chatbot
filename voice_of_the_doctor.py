#step 1a :Setup Text to Speech TTS model with  google TTS
import os 
from gtts import gTTS

def text_to_speech_with_gtts_old(input_text,output_filepath):
    language="en"

    audioobj=gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    return output_filepath 

input_text= "Hi this is AI with Shubham"
#text_to_speech_with_gtts_old(input_text=input_text,output_filepath="gtts_testing.mp3")



#step 1b:Setup Text to Speech TTS model with ElevenLAbs

import elevenlabs
from elevenlabs.client import ElevenLabs
from elevenlabs import save


from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEYS"))

ELEVENLAB_API_KEYS=os.environ.get("ELEVENLAB_API_KEYS")
def text_to_speech_with_elevenlabs_old(input_text, output_filepath):
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEYS"))
    
    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="JjpPU2Do2isL2c5DkxV2",  # replace with your voice ID if different
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128",
    )
    
    save(audio, output_filepath)
    return output_filepath 

#text_to_speech_with_elevenlabs_old(input_text,output_filepath="elevenlabs_testing.mp3")



#step 2: Use model for text output to voice

import subprocess
import platform

def text_to_speech_with_gtts(input_text,output_filepath):
    language="en"

    audioobj=gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    os_name=platform.system()
    try:
        if os_name=="Darwin": #MacOS
            subprocess.run(['afplay',output_filepath])
        elif os_name=="Windows": #Windows
            os.startfile(os.path.abspath(output_filepath)) 
            #abs_path = os.path.abspath(output_filepath) 
            #subprocess.run(['powershell','-c',f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
            #subprocess.run(['powershell','-c',f'Start-Process "{output_filepath}" -Wait'])
        elif os_name=="Linux": #Linux
            subprocess.run(['aplay',output_filepath]) #Alternative use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported Operating System")
    except Exception as e:
        print(f"An error occured whilw trying to play audio: {e}")

    return output_filepath 

input_text= "Hi this is AI with Shubham, Autoplay testing"
#text_to_speech_with_gtts(input_text=input_text,output_filepath="gtts_testing_autoplay.mp3")



def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEYS"))
    
    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="JjpPU2Do2isL2c5DkxV2",  # replace with your voice ID if different
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128",
    )
    
    save(audio, output_filepath)
    os_name=platform.system()
    try:
        if os_name=="Darwin": #MacOS
            subprocess.run(['afplay',output_filepath])
        elif os_name=="Windows": #Windows
            os.startfile(os.path.abspath(output_filepath)) 
            #abs_path = os.path.abspath(output_filepath) 
            #subprocess.run(['powershell','-c',f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
            #subprocess.run(['powershell','-c',f'Start-Process "{output_filepath}" -Wait'])
        elif os_name=="Linux": #Linux
            subprocess.run(['aplay',output_filepath]) #Alternative use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported Operating System")
    except Exception as e:
        print(f"An error occured whilw trying to play audio: {e}")
    return output_filepath 
#text_to_speech_with_elevenlabs(input_text,output_filepath="elevenlabs_testing_autoplay.mp3")
