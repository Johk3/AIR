from os import getenv
from openai import OpenAI
import numpy as np
import asyncio
import time
from threading import Thread
from qtpy.QtCore import QEvent, Qt, QTimer
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLineEdit,
    QTextEdit
)
from .napari_image_filters import (
    apply_grayscale,
    apply_saturation,
    apply_edge_enhance,
    apply_edge_detection,
    apply_gaussian_blur,
    apply_contrast_enhancement,
    apply_texture_analysis,
    apply_adaptive_threshold,
    apply_sharpening,
    apply_ridge_detection,
)
from .ai import GPT, ElevenLabsTTS
from .utils import run_tts_in_thread
# The exception handles the headless CICD testing
try:
    from .stt import STT
except OSError:
    STT = None
from .cellpose_launch import *

class ChatWidget(QWidget):
    """
    Multi-modal chat widget supporting text and speech interaction.
    Speech is triggered by holding the spacebar. (Not anymore, todo?)
    """

    def __init__(self, viewer, filter_widget):
        """
        Initialize the chat widget, available commands, and UI.
        """
        super().__init__()
        self.viewer = viewer
        self.cellpose_widget = None
        self.filter_widget = filter_widget  # Reference to an ImageFilterWidget, if needed
        self.setup_ui()
        self.available_commands = {
            "grayscale": apply_grayscale,
            "saturation": apply_saturation,
            "edge_enhance": apply_edge_enhance,
            "edge_detection": apply_edge_detection,
            "blur": apply_gaussian_blur,
            "contrast": apply_contrast_enhancement,
            "texture": apply_texture_analysis,
            "threshold": apply_adaptive_threshold,
            "sharpen": apply_sharpening,
            "ridge_detection": apply_ridge_detection,
        }

        voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 1,
            "use_speaker_boost": True
        }
        voice_id = "EXAVITQu4vr4xnSDxMaL"  # sarah
        model_id = "eleven_flash_v2_5"
        generation_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"

        AI_PROMPT = getenv("AI_PROMPT")
        OPENAI_API_KEY = getenv("OPENAI_API_KEY")
        ELEVENLABS_API_KEY = getenv("ELEVENLABS_API_KEY")

        self.ai_system_prompt = AI_PROMPT
        self.Chat = GPT(api_key=OPENAI_API_KEY, prompt=AI_PROMPT)
        self.Speak = ElevenLabsTTS(
            gen_uri=generation_url,
            api_key=ELEVENLABS_API_KEY,
            voice_settings=voice_settings)

        if STT is not None:
            self.stt = STT(api_key=OPENAI_API_KEY)

        # Stream mode variables
        self.streaming_active = False
        self.silence_timer = QTimer()
        self.silence_timer.timeout.connect(self.check_silence)
        self.silence_threshold = 2000  # 2 seconds of silence to trigger processing
        self.last_audio_level = 0
        self.silence_start_time = 0
        self.stream_transcript = ""

    def setup_ui(self):
        """Configure the widget's user interface."""
        layout = QVBoxLayout()

        # Chat history text area
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #cccccc;
                padding: 5px;
            }
            """
        )
        layout.addWidget(self.chat_history)

        # Horizontal layout for text input, send button, and record button
        input_layout = QHBoxLayout()

        # Input field for text messages
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(
            "Enter your image processing request...")
        self.input_field.setStyleSheet(
            """
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            """
        )
        self.input_field.returnPressed.connect(self.process_input)
        input_layout.addWidget(self.input_field)

        # Send button for text input
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0366d6;
                color: white;
                padding: 5px 15px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #0255b3;
            }
            """
        )
        self.send_button.clicked.connect(self.process_input)
        input_layout.addWidget(self.send_button)

        # Record button for audio input (press-and-hold)
        self.record_button = QPushButton("Hold to Record")
        self.record_button.setStyleSheet(
            """
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 5px 15px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            """
        )
        # When pressed, start recording; when released, stop and process the
        # audio.
        self.record_button.pressed.connect(self.start_audio_recording)
        self.record_button.released.connect(self.stop_audio_recording)
        input_layout.addWidget(self.record_button)

        # Stream button for continuous audio capture with auto-stop
        self.stream_button = QPushButton("Stream Mode")
        self.stream_button.setStyleSheet(
            """
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 5px 15px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked {
                background-color: #dc3545;
            }
            """
        )
        self.stream_button.setCheckable(True)
        self.stream_button.clicked.connect(self.toggle_streaming)
        input_layout.addWidget(self.stream_button)

        layout.addLayout(input_layout)
        self.setLayout(layout)

    def toggle_streaming(self):
        """Toggle between streaming and non-streaming modes"""
        if self.stream_button.isChecked():
            self.start_streaming()
        else:
            self.stop_streaming()

    def start_streaming(self):
        """Start streaming audio with automatic silence detection"""
        self.stream_button.setText("Stop Streaming")
        self.record_button.setEnabled(False)
        self.streaming_active = True
        self.stream_transcript = ""

        # Set up the streaming callback
        def handle_transcription(text):
            self.stream_transcript = text
            self.chat_history.append(f"[🎤] Listening: <i>{text}</i>")

        # Start streaming
        self.stt.start_streaming(callback=handle_transcription)

        # Start silence detection
        self.silence_start_time = time.time()
        self.silence_timer.start(500)  # Check every 500ms

        self.chat_history.append(
            "[🎤] Streaming started. I'll listen until you pause speaking...")

    def stop_streaming(self):
        """Stop streaming audio and process the final transcript"""
        if not self.streaming_active:
            return

        self.silence_timer.stop()
        self.stream_button.setText("Stream Mode")
        self.record_button.setEnabled(True)
        self.streaming_active = False

        # Stop the STT streaming
        self.stt.stop_streaming()

        # Process the final transcript if we have one
        if self.stream_transcript:
            self.process_transcript(self.stream_transcript)
            self.stream_transcript = ""

    def check_silence(self):
        """Check if there's been silence for the threshold duration"""
        if not self.streaming_active:
            return

        # Get current audio level (amplitude) from STT
        current_level = 0
        # if hasattr(self.stt, '_audio_data') and self.stt._audio_data:
        #     # Get the most recent audio chunk and calculate its RMS amplitude
        #     recent_chunk = self.stt._audio_data[-1]
        #     if recent_chunk.size > 0:
        #         print(np.mean(np.square(recent_chunk)))
        #         current_level = np.sqrt(np.mean(np.square(recent_chunk)))

        # If level is below threshold, consider it silence
        silence_threshold = 500  # Adjust based on your microphone sensitivity

        if current_level < silence_threshold:
            # If this is the start of silence
            if self.last_audio_level >= silence_threshold:
                self.silence_start_time = time.time()

            # Check if silence has lasted long enough and we have a transcript
            elapsed_silence = time.time() - self.silence_start_time
            if elapsed_silence > (
                    self.silence_threshold /
                    1000) and self.stream_transcript:
                # Auto-stop streaming and process
                self.stream_button.setChecked(False)
                self.stop_streaming()
        else:
            # Reset silence timer if there's sound
            self.silence_start_time = time.time()

        self.last_audio_level = current_level

    def start_audio_recording(self):
        """Start audio recording and update UI to indicate recording state."""
        self.record_button.setText("Recording...")
        self.stt.start_recording()

    def stop_audio_recording(self):
        """
        Stop audio recording, transcribe the audio, and process the transcript.
        Updates the UI and sends the transcribed text to the GPT API.
        """
        self.record_button.setText("Transcribing...")
        self.stt.stop_recording()
        transcript = self.stt.transcribe()
        self.process_transcript(transcript)
        self.record_button.setText("Hold to Record")

    def process_transcript(self, transcript):
        """Process a transcript from either recording or streaming"""
        self.chat_history.append(f"[👤] User: <i>{transcript}</i>")

        try:
            response_text, action = self.Chat.say(transcript)
            if response_text:
                self.add_to_chat(f'[🤖] <b>{response_text}</b>')
                Thread(
                    target=run_tts_in_thread,
                    args=(
                        self.Speak.stream_tts,
                        response_text),
                    daemon=True).start()
            if action:
                self.add_to_chat(f'[⚙️] <b>{self.format_action(action)}</b>')
                self.execute_command(action)
        except Exception as e:
            self.add_to_chat("[⚠️] Error: " + str(e))

    def process_input(self):
        """
        Handles User to AI interaction
        """
        user_input = self.input_field.text()
        if not user_input:
            return

        self.add_to_chat(f'[👤] User: <i>{user_input}</i>')
        self.input_field.clear()

        # Process with LLM
        try:
            response_text, action = self.Chat.say(user_input)
            if response_text:
                self.add_to_chat(f'[🤖] <b>{response_text}</b>')
                Thread(
                    target=run_tts_in_thread,
                    args=(
                        self.Speak.stream_tts,
                        response_text),
                    daemon=True).start()
            if action:
                self.add_to_chat(f'[⚙️] <b>{self.format_action(action)}</b>')
                self.execute_command(action)
        except Exception as e:
            self.add_to_chat("[⚠️] Error: " + str(e))

    def format_action(self, action):
        """
        str formating
        """
        return ", ".join([
            f"{act.action_name} {','.join(str(it) for it in act.action_args)}" for act in action
        ])

    def add_to_chat(self, message):
        """
        Appends to the chat a message
        """
        self.chat_history.append(message)

    def change_layer(self, curr_layer, filtered_array, filter_name):
        """
        Switches the current layer to the newest filtered layer
        """
        self.filter_widget._push_to_history(curr_layer)
        if hasattr(filtered_array, "__array__"):
            filtered_array = np.asarray(filtered_array)
        elif not isinstance(filtered_array, np.ndarray):
            filtered_array = np.array(filtered_array)

        # Create new layer with descriptive name and add to napari viewer
        new_layer_name = f"{curr_layer.name} | {filter_name}"
        self.viewer.add_image(filtered_array, name=new_layer_name)

    def _get_cellpose_widget(self):
        """Create (or return) the docked Cellpose widget and make sure it is visible."""
        if self.cellpose_widget is None:
            self.cellpose_widget = CellposeNapariWidget(self.viewer)
            # dock it on the right just once
            self.viewer.window.add_dock_widget(self.cellpose_widget,
                                            name="Cellpose", area="right")
        # if user closed the dock, reopen it
        elif self.cellpose_widget not in self.viewer.window._dock_widgets:
            self.viewer.window.add_dock_widget(self.cellpose_widget,
                                            name="Cellpose", area="right")
        return self.cellpose_widget

    def execute_command(self, command):
        """
        Executes a command sent by the LLM.
        """
        for act in command:

            # --- Cellpose entry point -----------------------------------------
            if act.action_name == "cell_segmentation":
                widget = self._get_cellpose_widget()

                model, chan1, diameter, flow_thr = list(act.action_args or [])
                widget.model_selector.setCurrentText(str(model))
                widget.chan1_selection.setCurrentIndex(chan1)
                widget.diameter_spin.setValue(diameter)
                widget.flow_spin.setValue(flow_thr)

                widget.run_segmentation()      # defaults already in the GUI
                widget.apply_to_napari()
                continue     

            # --- everything else (existing filters) ---------------------------
            func_name = act.action_name
            params    = act.action_args

            layer = self.filter_widget._get_current_layer()
            img   = self.filter_widget.original_data.copy()

            filtered = (
                self.available_commands[func_name](img, params[0])
                if params else self.available_commands[func_name](img)
            )
            self.change_layer(layer, filtered, func_name.title())
            if not params:
                self.filter_widget._push_to_history(layer)

    
    

