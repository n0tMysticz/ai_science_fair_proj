import os
import time
import cv2
import numpy as np
from pynput import keyboard
from picamera2 import Picamera2
from ai_edge_litert import interpreter as litert #interpreter compiles all of the data from the photo and describes the objects in simpler terms
import subprocess
import RPi.GPIO as GPIO

# --- DISCLAIMER - AI COLLABORATIVE DEBUGGING ---
# This script was developed by Tyler Huynh with assistance from Artifical Inteligence.
# I've created the design, logic flow, and import structure.
# While Generative AI Assistance (Gemini, Claude, Kimi, Github Copilot) was used for code debugging and library implemetation assistance.

# --- START AI ACKNOWLEDGEMENTS SECTION ---
# 1. Environment Setup & Troubleshooting (Lines 5-9):
#    - Gemini 3 Flash assisted in fixing Pylance import errors for 'cv2', 'pynput', 
#      and 'ai_edge_litert' inside the Raspberry Pi environment.
# 2. GPIO enabled (Lines 97-101)
#    - Claude 4.5 Sonnet helped with GPIO button support
# 3. AI detection (Lines 68-81 & 207-240):
#    - Gemini 3 Flash provided implementation details for LiteRT 
#      Interpreter (LiteRT), including tensor allocation (saving ram for the AI) and getting raw AI data into a useable format for this project.
# 4. Audio & Volume Control (Lines 111-125):
#    - Claude 4.5 Sonnet identified 'amixer' control failures and provided the correct 
#      shell commands for Raspberry Pi 'Speaker' and 'PCM' channels.
# 5. Piper TTS Integration (Lines 167-186 ):
#    - Claude 4.5 Sonnet helped structure the audio pipeline to route text 
#      through Piper and stream raw audio to 'aplay' or straight directly to the tts system where there is no audio latency.
# 6. Camera Preview & Rotation Logic (Lines 216-240):
#    - Claude 4.5 Sonnet debugged the 'Picamera2' capture loop and integrated 
#      OpenCV rotation constants for hardware-mounted camera adjustments.
# 7. Object Detection Logic (Lines 143-150, 280-285):
#    - Gemini 3 Flash suggested the logic for calculating object center-offsets 
#      to provide directional feedback (Left/Front/Right).
#    - Github Copilot assisted the debugging for OpenCV's rgb2bgr conversion.
# 5. Logging Camera Data and Results (lines 50, 239-230)
#    - Claude 4.5 Sonnet structured the logging to save images and log results.
# --- END AI ACKNOWLEDGEMENTS SECTION ---


# --- GPIO config ---
Camera_Button = 17
Volume_Button = 26

# path of the parts
main_folder = os.path.expanduser("~/ai_science_fair_proj")
cv_model = os.path.join(main_folder, "detect.tflite")
categories = os.path.join(main_folder, "labelmap.txt")
save_captures = os.path.join(main_folder, "captures")
detection_log = os.path.join(main_folder, "detection_log.txt")
cleanup = 25 # photos until deletion

if not os.path.exists(save_captures):
    os.makedirs(save_captures)

confidence_minimum = 0.55

camera_rotation = 0

# Piper TTS configuration
piper_path = os.path.join(main_folder, "piper", "piper") 
piper_model = os.path.join(main_folder, "voice_models", "voice.onnx")
use_tts = True  

class CVTesting:
    """
    Computer vision model for GPIO button control.
    """
    def __init__(self):
        print("Starting computer vision...")
        
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

        # variables for GPIO buttons
        self.is_ai_needed = False
        self.volume_pressed = False
        self.timer_start = 0
        self.hold_time = 0.5
        self.exit = False
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(Camera_Button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(Volume_Button, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(Camera_Button, GPIO.FALLING, callback=self.cam_activate, bouncetime=300)
        
        # esc key is still there 
        self.key_listener = keyboard.Listener(on_press=self.on_press)
        self.key_listener.start()

    def on_press(self, key):
        """Emergency escape key only."""
        try:
            if key == keyboard.Key.esc:
                self.exit = True
                return False
        except AttributeError:
            pass

    def cam_activate(self, channel):
        """GPIO button for AI capture."""
        self.is_ai_needed = True

    def check_volume_button(self):
        """Check volume button state and handle hold time."""
        current_state = GPIO.input(Volume_Button)
        
        if current_state == GPIO.LOW and not self.volume_pressed:
            self.volume_pressed = True
            self.timer_start = time.time()
        
        elif current_state == GPIO.HIGH and self.volume_pressed:
            duration = time.time() - self.timer_start
            self.handle_volume(duration)
            self.volume_pressed = False
            self.timer_start = 0

    def handle_volume(self, duration):
        """volume up/down based on button press duration."""
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
        
        try:
            if not os.path.exists(piper_path):
                print(f"Piper not found at: {piper_path}")
                return
            
            if not os.path.exists(piper_model):
                print(f"Voice model not found at: {piper_model}")
                return
            
            # outputs raw audio to aplay from piper tts
            process = subprocess.Popen(
                [piper_path, "--model", piper_model, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            audio_data, error = process.communicate(input=text.encode())
            
            if process.returncode != 0:
                print(f"Piper failed: {error.decode()}")
                return
            
            play_process = subprocess.Popen(
                ["aplay", "-D", "plughw:2,0", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            stdout, stderr = play_process.communicate(input=audio_data)
            
            if play_process.returncode != 0:
                print(f"aplay failed: {stderr.decode()}")
            else:
                print("TTS complete!")
            
        except Exception as e:
            print(f"[TTS ERROR] Exception: {e}")

    def analyze_frame(self, image):
        """Starts inference on a captured frame."""
        print("Analyzing Photo...")

        try:
            photos = sorted([f for f in os.listdir(save_captures) if f.startswith('capture_') and f.endswith('.jpg')])
            if len(photos) > cleanup:
                for old_photo in photos[:-cleanup]:
                    os.remove(os.path.join(save_captures, old_photo))
                print(f"Cleaned up {len(photos) - cleanup} old photos")
            
            image_prepared = cv2.resize(image, (self.input_width, self.input_height))
            input_tensor = np.expand_dims(image_prepared, axis=0)

            # post-image capture
            self.model.set_tensor(self.input_specs[0]['index'], input_tensor)
            self.model.invoke()

            boxes = self.model.get_tensor(self.output_specs[0]['index'])[0]
            classes = self.model.get_tensor(self.output_specs[1]['index'])[0]
            scores = self.model.get_tensor(self.output_specs[2]['index'])[0]
            num_detections = int(self.model.get_tensor(self.output_specs[3]['index'])[0])
            
            findings = []
            print("Top detections:")
            detection_details = []
            height, width = image.shape[:2]
            for i in range(min(num_detections, 10)):
                score = float(scores[i])
                class_id = int(classes[i])
                if 0 <= class_id < len(self.class_labels):
                    label = self.class_labels[class_id]
                    print(f"{label}: {score:.2f}")
                    detection_details.append(f"{label}: {score:.2f}")
                
                if score > confidence_minimum:
                    ymin, xmin, ymax, xmax = boxes[i]
                    cv2.rectangle(image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), (255, 0, 0), 2)
                    
                    center_x = float((xmin + xmax) / 2)
                    class_id = int(classes[i])
                    
                    if 0 <= class_id < len(self.class_labels):
                        label = self.class_labels[class_id]
                        direction = self.determine_location(center_x)
                        article = "an" if label.lower().startswith(('a', 'e', 'i', 'o', 'u')) else "a"
                        findings.append(f"{article} {label} {direction}")
            
            msg = f"I see: {' and '.join(findings)}" if findings else "No objects found were detected."
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(save_captures, f"capture_{timestamp}.jpg")
            cv2.imwrite(filename, image)
            print(f"  Photo saved to: {filename}")
            
            # Log findings to text file
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
            print("System Active.")
            print("-> GPIO Pin 17: Capture photo")
            print("-> GPIO Pin 26: Volume control (tap/hold)")
            print("-> Press ESC key for emergency exit")
            
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (640, 480)},
                controls={"AfMode": 2}
            )
            picam2.configure(config)
            picam2.start()
            
            print("Warming up camera...")
            time.sleep(2)
            
            window_name = "Camera Active"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            
            while True:
                try:
                    frame_rgb = picam2.capture_array()
                    frame = frame_rgb.copy()

                    if camera_rotation == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif camera_rotation == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif camera_rotation == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    cv2.putText(frame, "Camera: Active", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, frame)

                    # Check volume button state
                    self.check_volume_button()

                    if self.is_ai_needed:
                        self.is_ai_needed = False
                        
                        results = self.analyze_frame(frame)
                        msg = f"I see: {' and '.join(results)}" if results else "No objects found were detected."
                        print(f"{msg}")
                        
                        self.speak(msg)

                    # escape key exit
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or self.exit:
                        break
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Frame capture error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Failed to start camera: {e}")
        finally:
            print("Shutting down...")
            if picam2:
                picam2.stop()
            
            #clean up everything
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            GPIO.cleanup()
            self.key_listener.stop()

if __name__ == "__main__": # ignition key for the script, starts everything after everything was prepared
    test_suite = CVTesting()
    test_suite.start()