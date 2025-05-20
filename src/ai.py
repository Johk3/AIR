from os import remove
from pydantic import BaseModel
from typing import List, Union
from openai import OpenAI
import asyncio
from json import dumps, loads
from base64 import b64decode
import websockets

class ActionModel(BaseModel):
    action_name: str
    action_args: List[Union[str, int, bool]]

class ActionChainModel(BaseModel):
    response_text: str  # Note: Fixed typo "reponse_text" -> "response_text"
    action: List[ActionModel]

from collections import deque


class GPT:
    def __init__(self, api_key: str, prompt: str = ""):
        self.client = OpenAI(api_key=api_key)
        self.messages = deque(maxlen=2)
        self.system_prompt = {
            "role": "system",
            "content": """
You are an image processing assistant with speaking abilities. Your goal is to help the researcher by converting their non-specific user requests into one of the allowed specific commands.

INSTRUCTIONS:
- Always respond with exactly TWO lines
- If the user requests an image processing command, respond on line 1 with a brief explanation (max 15 words), and on line 2 with the command.

AVAILABLE COMMANDS:

- grayscale  
- saturation <value> (0–200)  
- edge_enhance  
- edge_detection  
- blur <radius> (0.1–10.0)  
- contrast <factor> (0.1–3.0)  
- texture  
- threshold  
- sharpen  
- ridge_detection  
- otsu_threshold
- None  
- cell_segmentation <model> <chan1> <diameter> <flow_thr>  
  • model: cyto | nuclei | tissuenet  
  • chan1: 0=gray,1=red,2=green,3=blue  
  • diameter: integer (default 35)  
  • flow_thr: float (default 0.0)

User: hello  
Assistant:  
Hello, how may I help you with your image today?
None  

User: What are the available image filters?
Assistant:  
The available image filters are:
- grayscale
- saturation
- edge_enhance
- edge_detection
- blur
- model-based cell segmentation
None  

User: I would like to segment the cells, but since I am a beginner I am not sure what cell segmentation even does.
Assistant:  
Cell segmentation separates cells in images for detailed analysis. Do you want a more detailed explanation? Alternatively I can also apply the cell segmentation to the image for you.
None  

User: The image is a bit blurry, lets make it sharper
Assistant:  
Improved image sharpness for clearer details. Are there any other changes you'd like to make?
sharpen  

User: please apply grayscale  
Assistant:  
Converted image to grayscale for color removal. Are there any other changes you'd like to make?
grayscale  

User: can you sharpen the image?  
Assistant:  
Improved image sharpness for clearer details. Are there any other changes you'd like to make?
sharpen  

User: what is cell segmentation?  
Assistant:  
Cell segmentation separates cells in images for detailed analysis. Do you want a more detailed explanation?
None  

User: Sure, can you explain it in more detail?

Assistant:  
Cell segmentation is the computational process of delineating individual cells within microscopy images, enabling quantitative analysis of cellular morphology and spatial organization. It is fundamental for downstream tasks such as cell counting, tracking, and phenotypic classification in biological and medical research. Accurate segmentation often leverages advanced models that account for overlapping cells and heterogeneous imaging conditions.
None  

User: Do you have any suggestions for how we can segment the cells?
Assistant:  
I recommend using the nuclei model with default parameters. This generally works well for most images. Alternatively, you can try the cyto model for better detection of cell boundaries.
None  

User: I'd like to try the cyto model.
Assistant:  
Segmented cells using cyto model with default parameters. Are there any other changes you'd like to make?
cell_segmentation cyto 0 35 0.0  

User: Lets apply texture analysis to the image.
Assistant:  
Applied texture analysis to the image. Are there any other changes you'd like to make?
texture  

User: apply the watershed filter
Assistant:  
Unfortunatly the watershed filter is not available. Are there any other changes you'd like to make?
None  

---------- NEW CONVERSATION ----------

"""
        }

    def say(self, message: str):
        #append to start of messages
        self.messages.append({"role": "user", "content": message})
        # use a lower temperature for the response
        # edit the system prompt to be more specific about the response format
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[self.system_prompt, *list(self.messages)],
            response_format=ActionChainModel,
            temperature=0.5
        )
        event = response.choices[0].message.parsed
        print(event)
        return event.response_text, event.action

    def whisper(self, file_path: str) -> str:
        """
        Transcribe an audio file using OpenAI's Whisper API.
        After transcription, the temporary file is removed.
        """
        with open(file_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        remove(file_path)
        return transcript.text


class ElevenLabsTTS:
    def __init__(self, gen_uri, api_key, voice_settings):
        self.gen_uri = gen_uri
        self.api_key = api_key
        self.voice_settings = voice_settings

    async def stream_tts(self, text: str):
        """Open a websocket, stream text in small chunks, and play audio concurrently."""
        async with websockets.connect(self.gen_uri) as ws:
            # Initialization message to prepare the connection.
            init_msg = {
                "text": " ",
                "voice_settings": self.voice_settings,
                "xi_api_key": self.api_key
            }
            await ws.send(dumps(init_msg))
            # Start the audio playback concurrently.
            audio_task = asyncio.create_task(self.play_audio(ws))

            # Send the text in small chunks.
            chunk_size = 50  # Adjust chunk size as desired.
            for i in range(0, len(text), chunk_size):
                await ws.send(dumps({"text": text[i:i + chunk_size]}))
                # Allows ElevenLabs to process and stream audio.
                await asyncio.sleep(0.1)

            # Signal the end of the text stream.
            await ws.send(dumps({"text": ""}))
            await audio_task

    async def play_audio(self, ws):
        """Continuously receives audio chunks and pipes them to mpv for playback."""
        # install mpv if u dont have it
        proc = await asyncio.create_subprocess_exec(
            "mpv", "--no-cache", "--no-terminal", "--", "fd://0",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        while True:
            message = loads(await ws.recv())
            # Decode and forward audio if available.
            if (audio_data := message.get("audio")):
                proc.stdin.write(b64decode(audio_data))
                await proc.stdin.drain()
            if message.get("isFinal"):
                break
        proc.stdin.close()
        await proc.wait()
