#!/usr/bin/env python3
"""
Detect Status - Human Posture Detection System based on Moondream
Takes photos every minute and analyzes human posture: sitting, standing, away from seat
Implements 30-minute sitting alert and 5-minute standing/away reset logic
"""

import cv2
import requests
import base64
import time
import os
import json
import threading
import subprocess
from datetime import datetime, timedelta
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque
import numpy as np

class PersonStatus(Enum):
    """Human posture status enumeration"""
    SITTING = "person sitting"
    STANDING = "person standing" 
    AWAY = "person away from seat"

class DetectStatus:
    def __init__(self):
        # Status tracking
        self.current_status = PersonStatus.SITTING
        self.status_history = deque(maxlen=720)  # Save 24 hours of data (one point per 2 minutes)
        self.sitting_start_time = None
        self.standing_away_start_time = None
        self.last_alert_time = None
        
        # Configuration parameters
        self.SITTING_ALERT_MINUTES = 30  # Alert after sitting for 30 minutes
        self.STANDING_AWAY_RESET_MINUTES = 4  # Reset timer after standing/away for 4 minutes
        
        # Create image storage directory
        self.image_dir = "detection_images"
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Create log file and reset it
        self.log_file = "detection_log.txt"
        self.reset_log_file()
        
        # Initialize plot
        self.setup_plot()
        
    def reset_log_file(self):
        """Reset log file at startup"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("")  # Clear the file
            print(f"Log file reset: {self.log_file}")
        except Exception as e:
            print(f"Error resetting log file: {e}")
        
    def warm_up_model(self):
        """Warm up the models by testing the full pipeline"""
        print("Warming up models...")
        try:
            # Capture a test image
            print("Taking a test photo...")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                print("Error: Could not open camera")
                return False
            
            print("Camera warming up...")
            time.sleep(2)
            for i in range(30):
                cap.read()
            
            print("Capturing test image...")
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("Error: Could not capture image")
                return False
            
            # Save test image
            capture_time = datetime.now()
            timestamp = capture_time.strftime("%Y%m%d_%H%M%S")
            test_filename = f"{self.image_dir}/warmup_{timestamp}.jpg"
            cv2.imwrite(test_filename, frame)
            print(f"Test image saved: {test_filename}")
            
            # Test the full pipeline
            print("Testing AI pipeline...")
            ai_response = self.analyze_with_two_calls(test_filename)
            
            if ai_response:
                parsed_status = self.parse_status(ai_response)
                if parsed_status:
                    # Log warmup result
                    timestamp = capture_time.strftime("%Y-%m-%d %H:%M:%S")
                    self.log_status(timestamp, parsed_status)
                    print(f"SUCCESS: Models warmed up successfully - detected: {parsed_status.value}")
                    print(f"Warmup image kept: {test_filename}")
                    return True
            
            print("FAILED: Model warm-up failed")
            return False
                
        except Exception as e:
            print(f"FAILED: Model warm-up failed with error: {e}")
            return False

    def capture_image(self):
        """Capture image from webcam using OpenCV (same as moondream_simple)"""
        print("Opening camera...")
        
        # Open webcam (same as infer.py and hand_pose.py)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
        print("Camera warming up...")
        # Let camera warm up - need more time for proper exposure
        time.sleep(2)
        for i in range(30):
            cap.read()
        
        # Capture frame
        print("Smile! Capturing in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("*CLICK*")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not capture image")
            return None
        
        # Save image with timestamp
        capture_time = datetime.now()
        timestamp = capture_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.image_dir}/detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved as: {filename}")
        return filename, capture_time

    def analyze_with_two_calls(self, image_path):
        """Method 2: Two-step approach - Moondream describes, Ollama classifies"""
        # Step 1: Ask Moondream to describe the image
        description = self._query_moondream(image_path, "Describe what you see in this image.")
        
        if not description:
            return None
        
        # Step 2: Send description to Ollama for posture classification
        classification_result = self._query_ollama_text_only(f"""Based on this description: "{description}"

Answer with ONLY the number:

1 = person sitting
2 = person standing  
3 = person away from seat

Answer:""")
        
        return classification_result

    def _query_moondream(self, image_path, prompt):
        """Ask Moondream about the image with streaming response (same as moondream_simple)"""
        
        # Encode image to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        print(f"\nAsking Moondream: {prompt}")
        print("\nMoondream: ", end="", flush=True)
        
        try:
            # Query Moondream with streaming
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "moondream:latest",
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": True
                },
                timeout=300,  # Increased timeout to 5 minutes
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            import json
                            chunk = json.loads(line)
                            token = chunk.get('response', '')
                            print(token, end="", flush=True)
                            full_response += token
                        except json.JSONDecodeError as e:
                            print(f"\n[JSON Error] {e}")
                            continue
                
                print("\n")  # New line after response
                if full_response.strip():
                    return full_response
                else:
                    print("[WARNING] Empty response received")
                    return None
            else:
                print(f"\nError: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print("\n[TIMEOUT] Moondream is taking too long. The model might be processing a large image.")
            print("Tip: Try using a smaller image or wait for the model to finish loading.")
            return None
    def _query_ollama_text_only(self, prompt):
        """Query Ollama with text-only prompt (no image)"""
        print(f"\nAsking Ollama: {prompt}")
        print("\nOllama: ", end="", flush=True)
        
        try:
            # Query Ollama with streaming and stable parameters
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "phi3:mini",  # Use phi3:mini for text-only queries
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.1,  # Very low temperature for stable output
                        "top_p": 0.1,        # Low top_p for focused responses
                        "repeat_penalty": 1.1  # Prevent repetition
                    }
                },
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            import json
                            chunk = json.loads(line)
                            token = chunk.get('response', '')
                            print(token, end="", flush=True)
                            full_response += token
                        except json.JSONDecodeError as e:
                            print(f"\n[JSON Error] {e}")
                            continue
                
                print("\n")  # New line after response
                if full_response.strip():
                    return full_response
                else:
                    print("[WARNING] Empty response received")
                    return None
            else:
                print(f"\nError: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print("\n[TIMEOUT] Ollama is taking too long.")
            return None
        except Exception as e:
            print(f"\n[ERROR] {e}")
            return None

    def parse_status(self, ai_response):
        """Parse AI response and extract status"""
        if not ai_response:
            return None
            
        response_lower = ai_response.lower().strip()
        
        # First check for pure numbers: "1", "2", "3"
        if response_lower == "1":
            return PersonStatus.SITTING
        elif response_lower == "2":
            return PersonStatus.STANDING
        elif response_lower == "3":
            return PersonStatus.AWAY
        
        # Look for exact format: "1. person sitting", "2. person standing", "3. person away from seat"
        if "1. person sitting" in response_lower or "1 person sitting" in response_lower:
            return PersonStatus.SITTING
        elif "2. person standing" in response_lower or "2 person standing" in response_lower:
            return PersonStatus.STANDING
        elif "3. person away from seat" in response_lower or "3 person away from seat" in response_lower:
            return PersonStatus.AWAY
        
        # Fallback: look for keywords
        if "sitting" in response_lower:
            return PersonStatus.SITTING
        elif "standing" in response_lower:
            return PersonStatus.STANDING
        elif "away" in response_lower:
            return PersonStatus.AWAY
        else:
            print(f"Cannot parse status: {ai_response}")
            return None

    def update_status_logic(self, new_status):
        """Update status logic and handle alerts"""
        current_time = datetime.now()
        alert_triggered = False
        
        if new_status == PersonStatus.SITTING:
            # If returning to sitting from standing/away, reset timer
            if self.current_status in [PersonStatus.STANDING, PersonStatus.AWAY]:
                if self.standing_away_start_time:
                    away_duration = current_time - self.standing_away_start_time
                    if away_duration.total_seconds() / 60 <= self.STANDING_AWAY_RESET_MINUTES:
                        print(f"Standing/away time {away_duration.total_seconds()/60:.1f} minutes, not exceeding {self.STANDING_AWAY_RESET_MINUTES} minutes, not resetting sitting timer")
                    else:
                        print(f"Standing/away time {away_duration.total_seconds()/60:.1f} minutes, exceeding {self.STANDING_AWAY_RESET_MINUTES} minutes, resetting sitting timer")
                        self.sitting_start_time = current_time
                else:
                    self.sitting_start_time = current_time
            elif self.current_status == PersonStatus.SITTING:
                # Continue sitting, check if alert needed
                if self.sitting_start_time:
                    sitting_duration = current_time - self.sitting_start_time
                    if sitting_duration.total_seconds() / 60 >= self.SITTING_ALERT_MINUTES:
                        # Check if already alerted (avoid duplicate alerts)
                        if not self.last_alert_time or (current_time - self.last_alert_time).total_seconds() >= 60:
                            self.play_alert()
                            self.last_alert_time = current_time
                            alert_triggered = True
            else:
                self.sitting_start_time = current_time
                
        elif new_status in [PersonStatus.STANDING, PersonStatus.AWAY]:
            # Start standing or away
            if self.current_status == PersonStatus.SITTING:
                self.standing_away_start_time = current_time
                print(f"Starting standing/away, timer started")
        
        self.current_status = new_status
        
        # Record status history
        self.status_history.append({
            'timestamp': current_time,
            'status': new_status,
            'alert': alert_triggered
        })
        
        return alert_triggered

    def play_alert(self):
        """Play audio alert"""
        print(f"\nALERT: You have been sitting for more than {self.SITTING_ALERT_MINUTES} minutes, please get up and move around!")
        
        # Use espeak to play English alert
        try:
            subprocess.run(['espeak', 'You have been sitting for more than 30 minutes, please get up and move around'], 
                         check=False, timeout=5)
        except:
            # If espeak is not available, use system beep
            try:
                subprocess.run(['beep'], check=False, timeout=2)
            except:
                print("Cannot play audio alert")

    def setup_plot(self):
        """Setup plot"""
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_title('Human Posture Detection Timeline', fontsize=14)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Status')
        
        # Set y-axis labels
        status_labels = ['Sitting', 'Standing', 'Away']
        self.ax.set_yticks([0, 1, 2])
        self.ax.set_yticklabels(status_labels)
        self.ax.set_ylim(-0.5, 2.5)

    def update_plot(self):
        """Update plot"""
        if not self.status_history:
            return
            
        # Prepare data
        timestamps = [item['timestamp'] for item in self.status_history]
        status_values = []
        alert_flags = []
        
        for item in self.status_history:
            if item['status'] == PersonStatus.SITTING:
                status_values.append(0)
            elif item['status'] == PersonStatus.STANDING:
                status_values.append(1)
            else:  # AWAY
                status_values.append(2)
            alert_flags.append(item['alert'])
        
        # Clear old plot
        self.ax.clear()
        
        # Draw status points
        colors = ['green', 'blue', 'red']  # sitting-green, standing-blue, away-red
        for i, (ts, status_val) in enumerate(zip(timestamps, status_values)):
            color = colors[status_val]
            self.ax.scatter(ts, status_val, c=color, s=50, alpha=0.7)
            
            # If there's an alert, add alert marker
            if alert_flags[i]:
                self.ax.scatter(ts, status_val, c='yellow', s=100, marker='*', alpha=0.8)
        
        # Set plot
        self.ax.set_title('Human Posture Detection Timeline', fontsize=14)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Status')
        self.ax.set_yticks([0, 1, 2])
        self.ax.set_yticklabels(['Sitting', 'Standing', 'Away'])
        self.ax.set_ylim(-0.5, 2.5)
        
        # Format x-axis time display
        if len(timestamps) > 1:
            time_range = timestamps[-1] - timestamps[0]
            if time_range.total_seconds() > 3600:  # More than 1 hour
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            else:
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def log_status(self, timestamp, status, alert=False):
        """Log status to txt file - time, status, and alert if triggered"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if alert:
                    log_entry = f"{timestamp} - {status.value} - ALERT\n"
                else:
                    log_entry = f"{timestamp} - {status.value}\n"
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def run_detection_loop(self):
        """Run detection loop"""
        print(f"\nStarting detection loop...")
        print("Using two-step method: Moondream describes image, Ollama classifies posture")
        print("Detection interval: 2 minutes")
        print("Press Ctrl+C to stop detection")
        
        try:
            while True:
                # Capture image
                result = self.capture_image()
                if not result:
                    print("Skipping this detection")
                    time.sleep(120)
                    continue
                
                image_path, capture_time = result
                
                # AI analysis using two-step method
                ai_response = self.analyze_with_two_calls(image_path)
                
                if ai_response:
                    print(f"AI analysis result: {ai_response}")
                    
                    # Parse status
                    new_status = self.parse_status(ai_response)
                    if new_status:
                        print(f"Parsed status: {new_status.value}")
                        
                        # Update status logic
                        alert_triggered = self.update_status_logic(new_status)
                        
                        # Log status to txt file using capture time
                        timestamp = capture_time.strftime("%Y-%m-%d %H:%M:%S")
                        self.log_status(timestamp, new_status, alert_triggered)
                        
                        # Update plot
                        self.update_plot()
                        
                        if alert_triggered:
                            print("ALERT triggered!")
                    else:
                        print("Status parsing failed")
                else:
                    print("AI analysis completely failed, skipping this detection")
                
                # Wait for next detection (2 minutes)
                print(f"Waiting for next detection (2 minutes)...")
                time.sleep(120)
                
        except KeyboardInterrupt:
            print("\nDetection stopped")
            plt.close()

def restart_ollama_service():
    """Restart Ollama service asynchronously for better performance"""
    print("Restarting Ollama service asynchronously...")
    
    def restart_async():
        """Async restart function"""
        try:
            # Kill existing Ollama processes (non-blocking)
            subprocess.run(['sudo', 'pkill', '-f', 'ollama'], 
                          capture_output=True, text=True)
            print("Stopped existing Ollama processes")
            
            # Wait for processes to stop
            time.sleep(3)
            
            # Start Ollama service in background
            subprocess.Popen(['ollama', 'serve'], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
            print("Started Ollama service")
            
            # Wait for service to start
            time.sleep(5)
            
            # Pull required models
            print("Pulling required models...")
            subprocess.run(['ollama', 'pull', 'moondream:latest'], 
                          capture_output=True, text=True)
            print("Pulled moondream:latest")
            
            subprocess.run(['ollama', 'pull', 'phi3:mini'], 
                          capture_output=True, text=True)
            print("Pulled phi3:mini")
            
            # Test if service is running
            for attempt in range(3):
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code == 200:
                        print("SUCCESS: Ollama service restarted successfully")
                        return True
                except:
                    if attempt < 2:
                        print(f"Attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)
                    else:
                        print("FAILED: Could not restart Ollama service")
                        return False
            
        except Exception as e:
            print(f"Error restarting Ollama: {e}")
            return False
    
    # Start restart in background thread
    restart_thread = threading.Thread(target=restart_async, daemon=True)
    restart_thread.start()
    
    # Wait a bit for the restart to begin
    time.sleep(2)
    print("Ollama restart initiated in background...")
    
    return True

def main():
    """Main function"""
    print("DetectStatus - Human Posture Detection System")
    print("=" * 50)
    
    # Start Ollama restart in background (non-blocking)
    restart_ollama_service()
    
    # Create detection system while Ollama is restarting
    print("Creating detection system...")
    detector = DetectStatus()
    
    # Wait a bit more for Ollama to be ready
    print("Waiting for Ollama service to be ready...")
    time.sleep(8)  # Total wait time: 2 (from restart) + 8 = 10 seconds
    
    # Warm up the model
    print("Warming up AI models...")
    if not detector.warm_up_model():
        print("Model warm-up failed. Please check Ollama service and try again.")
        return
    
    # Start detection (using two-step method)
    detector.run_detection_loop()

if __name__ == "__main__":
    main()
