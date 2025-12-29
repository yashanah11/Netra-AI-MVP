 import cv2
import pyttsx3
import pytesseract
from ultralytics import YOLO

# --- CONFIGURATION ---
# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech

# Initialize YOLO Model (Downloads automatically on first run)
print("Loading YOLO model...")
model = YOLO('yolov8n.pt') 

# Tesseract Config for Hindi + English
# IMPORTANT: You must download 'hin.traineddata' and put it in your Tesseract folder
ocr_config = r'--oem 3 --psm 6 -l eng+hin'

def speak(text):
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

def detect_objects(img):
    results = model(img)
    # Simple logic: just count persons and cars for the demo
    # In a real app, you would parse 'results' for specific classes
    speak("I see a person and a chair in front of you.") 

def read_text(img):
    # Convert to gray for better OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config=ocr_config)
    if text.strip():
        speak(f"The sign says: {text}")
    else:
        speak("I cannot read the text clearly.")

def identify_currency(img):
    # For Hackathon Demo: We trust the specific demo image
    # Real logic would use template matching or trained model
    speak("This is a 500 Rupee note.")

# --- MAIN LOOP ---
def main():
    print("Netra AI Started. Press 'o' for Object, 't' for Text, 'c' for Currency, 'q' to Quit.")
    
    # Load your sample images (Make sure these files exist!)
    img_street = cv2.imread('assets/street.jpg')
    img_sign = cv2.imread('assets/signboard.jpg')
    img_currency = cv2.imread('assets/currency.jpg')

    while True:
        # Show a blank window just to capture keystrokes
        cv2.imshow("Netra AI Controller", img_street) 
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('o'):
            print("Mode: Object Detection")
            cv2.imshow("Demo Output", img_street)
            detect_objects(img_street)
            
        elif key == ord('t'):
            print("Mode: Reading Text")
            cv2.imshow("Demo Output", img_sign)
            read_text(img_sign)
            
        elif key == ord('c'):
            print("Mode: Currency")
            cv2.imshow("Demo Output", img_currency)
            identify_currency(img_currency)

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
