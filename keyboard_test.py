import os
import time
import cv2
import numpy as np
from pynput import keyboard
from picamera2 import Picamera2
from ai_edge_litert import interpreter as litert
import subprocess

# --- DISCLAIMER - AI COLLABORATIVE DEBUGGING ---
# This script was developed by Tyler Huynh with assistance from Artifical Inteligence.
# I've created the design, logic flow, and import structure.
# While Generative AI Assistance (Gemini, Claude, Kimi, Github Copilot) was used for code debugging and library implemetation assistance.

# --- START AI ACKNOWLEDGEMENTS SECTION ---
# 1. Environment & Imports:
#    - Gemini 3 Fast resolved Pylance environment issues for 'cv2' and 'ai_edge_litert' and
#    - Implemented 'pynput' keyboard listeners to simulate GPIO button hardware events.
# 2. Computer Vision & Inference (Lines 57-70, 186-226):
#    - Gemini 3 Fast integrated LiteRT (TFLite) interpreter for on-device inference.
#    - Claude 4.5 Sonnet assisted in mapping model output tensors to readable label arrays.
# 3. Audio & TTS Pipeline (Lines 130-180):
#    - Claude 4.5 Sonnet debugged 'amixer' shell commands for USB audio card channel management (-c 2).
#    - Gemini 3 Fast and Claude 4.5 Sonnet structured the audio pipeline to stream Piper TTS text-to-speech to 'aplay'.
# 4. Image Processing & Logic (Lines 129-136, 261-276):
#    - GitHub Copilot provided direction for spatial object detection (Left/Right/Front).
#    - Claude 4.5 Sonnet assisted with OpenCV frame rotation and UI text overlays.
# 5. Logging Camera Data and Results (lines 38, 248-255)
#    - Claude 4.5 Sonnet structured the logging to save images and log results.
# --- END AI ACKNOWLEDGEMENTS SECTION ---

# --- test config for keyboard ---
# path of the parts
main_folder = os.path.expanduser("~/ai_science_fair_proj")
cv_model = os.path.join(main_folder, "detect.tflite")
categories = os.path.join(main_folder, "labelmap.txt")
save_captures = os.path.join(main_folder, "captures")
detection_log = os.path.join(main_folder, "detection_log.txt") 

if not os.path.exists(save_captures):
    os.makedirs(save_captures)

confidence_minimum = 0.5

# Camera rotation (0, 90, 180, or 270 degrees)
camera_rotation = 0

# Piper TTS configuration
piper_path = os.path.join(main_folder, "piper", "piper")  # Path to piper executable
piper_model = os.path.join(main_folder, "voice_models", "en_US-lessac-medium.onnx")
use_tts = True  # Set to False to disable voice output

class CVTesting:
    """
    Simulated computer vision model for keyboard testing.
    """
    def __init__(self):
        print("[Simulation] Starting computer vision!...")
        
        try:
            self.model = litert.Interpreter(model_path=cv_model)
            self.model.allocate_tensors()
            print(f"Model loaded successfully from {cv_model}")
        except Exception as error:
            print(f"Failed to initialize model: {error}")
            print("Check if the model is in the project folder.")
            exit(1)
            
        # prerequisites (gets data from the model)
        self.input_specs = self.model.get_input_details()
        self.output_specs = self.model.get_output_details()
        self.input_height = self.input_specs[0]['shape'][1]
        self.input_width = self.input_specs[0]['shape'][2]
    
        try:
            with open(categories, 'r') as f:
                self.class_labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"{categories} not found.")
            exit(1)

        # variables based off of keybinds
        self.is_ai_needed = False
        self.volume_pressed = False
        self.timer_start = 0
        self.hold_time = 0.5
        self.exit = False

        # keypress checker
        self.key_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.key_listener.start()

    def on_press(self, key):
        """just simulates my gpio buttons for keyboard."""
        try:
            if key == keyboard.Key.space:
                self.is_ai_needed = True
            
            if key == keyboard.Key.esc:
                self.exit = True
                return False
            
            if key == keyboard.Key.up and not self.volume_pressed:
                self.volume_pressed = True
                self.timer_start = time.time()
        except AttributeError:
            pass

    def on_release(self, key):
        """captures key up event for interaction time."""
        try:
            if key == keyboard.Key.up:
                duration = time.time() - self.timer_start
                self.keyboard_volume(duration)
                self.volume_pressed = False
                self.timer_start = 0
        except AttributeError:
            pass

    def keyboard_volume(self, duration):
        """volume up/down with a keypress."""
        if duration < self.hold_time:
            print(f"Volume up (Duration: {duration:.2f}s)")
            os.system("amixer -c 2 sset Speaker 1%+ 2>/dev/null || amixer -c 2 sset PCM 1%+ 2>/dev/null")
        else:
            print(f"Volume down (Duration: {duration:.2f}s)")
            os.system("amixer -c 2 sset Speaker 5%- 2>/dev/null || amixer -c 2 sset PCM 5%- 2>/dev/null")

    def determine_location(self, x_center_offset):
        """turns object's direction to a string."""
        if x_center_offset < 0.33: 
            return "on your left"
        elif x_center_offset > 0.66: 
            return "on your right"
        else: 
            return "in front of you"
    
    def speak(self, text):
        """Use Piper TTS to speak text."""
        if not use_tts:
            return
        
        print(f"[TTS] Attempting to speak: {text}")
        
        try:
            # Check if piper exists
            if not os.path.exists(piper_path):
                print(f"[TTS ERROR] Piper not found at: {piper_path}")
                return
            
            if not os.path.exists(piper_model):
                print(f"[TTS ERROR] Voice model not found at: {piper_model}")
                return
            
            print("[TTS] Generating speech...")
            # Use piper to generate speech and play with aplay
            process = subprocess.Popen(
                [piper_path, "--model", piper_model, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            audio_data, error = process.communicate(input=text.encode())
            
            if process.returncode != 0:
                print(f"[TTS ERROR] Piper failed: {error.decode()}")
                return
            
            print(f"[TTS] Playing audio ({len(audio_data)} bytes)...")
            # Play the audio with aplay on card 2 (USB audio)
            play_process = subprocess.Popen(
                ["aplay", "-D", "plughw:2,0", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = play_process.communicate(input=audio_data)
            
            if play_process.returncode != 0:
                print(f"[TTS ERROR] aplay failed: {stderr.decode()}")
            else:
                print("[TTS] Speech complete")
            
        except Exception as e:
            print(f"[TTS ERROR] Exception: {e}")

    def analyze_frame(self, image):
        """Starts inference on a captured frame."""
        print("Analyzing Photo...")

        try:
            # Clean up old photos if more than 50 exist
            photos = sorted([f for f in os.listdir(save_captures) if f.startswith('capture_') and f.endswith('.jpg')])
            if len(photos) > 50:
                # Delete oldest photos to keep only 50
                for old_photo in photos[:-50]:
                    os.remove(os.path.join(save_captures, old_photo))
                print(f"Cleaned up {len(photos) - 50} old photos")
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(save_captures, f"capture_{timestamp}.jpg")
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, image_bgr)
            print(f"Photo saved to: {filename}")
            
            image_prepared = cv2.resize(image, (self.input_width, self.input_height))
            input_tensor = np.expand_dims(image_prepared, axis=0)

            # post-image capture
            self.model.set_tensor(self.input_specs[0]['index'], input_tensor)
            self.model.invoke()

            boxes = self.model.get_tensor(self.output_specs[0]['index'])[0]
            classes = self.model.get_tensor(self.output_specs[1]['index'])[0]
            scores = self.model.get_tensor(self.output_specs[2]['index'])[0]
            num_detections = int(self.model.get_tensor(self.output_specs[3]['index'])[0])
            
            # Print and log top detections with confidence scores
            print("Top detections:")
            detection_details = []
            for i in range(min(num_detections, 10)):
                score = float(scores[i])
                class_id = int(classes[i])
                if 0 <= class_id < len(self.class_labels):
                    label = self.class_labels[class_id]
                    print(f"{label}: {score:.2f}")
                    detection_details.append(f"{label}: {score:.2f}")
            
            findings = []
            for i in range(min(num_detections, 10)):
                score = float(scores[i])
                
                if score > confidence_minimum:
                    ymin, xmin, ymax, xmax = boxes[i]
                    
                    center_x = float((xmin + xmax) / 2)
                    class_id = int(classes[i])
                    
                    if 0 <= class_id < len(self.class_labels):
                        label = self.class_labels[class_id]
                        direction = self.determine_location(center_x)
                        findings.append(f"{label} {direction}")
            
            # logs results to txt file
            msg = f"I see: {', '.join(findings)}" if findings else "No objects found."
            log_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{log_timestamp}] {msg}\n"
            if detection_details:
                log_message += f"  Confidence scores: {', '.join(detection_details)}\n"
            with open(detection_log, 'a') as log_file:
                log_file.write(log_message)
            
            return findings
        except Exception as e:
            print(f"Frame analysis failed: {e}")
            return []

    def start(self):
        """Main loop for video capture and UI feedback."""
        picam2 = None
        try:
            print("Testing Active.")
            print("-> Press spacebar to capture a scene.")
            print("-> Tap/Hold up arrow to test volume.")
            print("-> Press escape to exit testing.")
            
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (640, 480)},
                controls={"AfMode": 2}
            )
            picam2.configure(config)
            picam2.start()
            
            print("Warming up camera...")
            time.sleep(2)
            
            window_name = "Camera Activated - Testing"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            
            while True:
                try:
                    frame_rgb = picam2.capture_array()
                    frame = frame_rgb.copy()
                    
                    # Apply rotation if needed
                    if camera_rotation == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif camera_rotation == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif camera_rotation == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    cv2.putText(frame, "Camera: Active", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, frame)

                    if self.is_ai_needed:
                        self.is_ai_needed = False
                        
                        results = self.analyze_frame(frame)
                        msg = f"I see: {', '.join(results)}" if results else "No objects found."
                        print(f"{msg}")
                        
                        # Speak the results
                        self.speak(msg)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or self.exit:
                        print("Test terminated")
                        break
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Frame capture error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Failed to start camera: {e}")
        finally:
            print("Quitting Camera")
            if picam2:
                picam2.stop()
            
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self.key_listener.stop()

if __name__ == "__main__":
    test_suite = CVTesting()
    test_suite.start()