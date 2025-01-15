from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder

import os
from tkinter import filedialog, Tk


from VGG_Inference import VGG16Inference

Builder.load_file(r'C:\Users\user\Desktop\psh\project\KIVY\VGG_classification_app\app_layout.kv')

class ClassificationApp(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.ids.filechooser.path = r'C:\Users\user\Desktop\psh\project\KIVY\VGG_classification_app\test_data'
        self.label = None
        self.cracked_prob = None
        self.uncracked_prob = None
        self.file_path = None 
        
        model_path = r"C:/Users/user/Desktop/psh/project/KIVY/VGG_classification_app/vgg16_weights.pth"
        classes = ['cracked', 'uncracked']
        self.vgg_inference = VGG16Inference(model_path=model_path, classes=classes)
        
    def select_image(self):
        # Hide Tkinter root window
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Open file dialog for image selection
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*"))
        )
        print(f"Selected file path: {file_path}")

        # Validate and process the selected file
        if file_path and os.path.exists(file_path):
            self.ids.my_img.source = file_path  # Update image view
            self.select_inf(file_path)  # Pass to select_inf
        else:
            print("Invalid file selected or file does not exist.")
         
                
    def select_inf(self, filename):
        # Handle both FileChooserIconView and direct file path
        if isinstance(filename, list) and filename:  # FileChooserIconView
            selected_file = filename[0]  # First selected file
            self.inference(selected_file)
        elif isinstance(filename, str) and os.path.exists(filename):  # Direct file path
            self.inference(filename)
        else:
            print("Invalid file or no file selected.")
            

    def selected(self, filename) :
        try:
            self.ids.my_img.source = filename[0]
            print(filename)
        except:
            pass
    
    def inference(self, img_path):
        self.label, self.cracked_prob, self.uncracked_prob = self.vgg_inference.predict(img_path)
        
        # UI 업데이트
        self.ids.my_label.text = f'Inference: {self.label}'
        self.ids.cracked.text = f'Cracked: {self.cracked_prob:.2f} %'
        self.ids.uncracked.text = f'Uncracked: {self.uncracked_prob:.2f} %'

        # 선택한 이미지 표시
        self.ids.my_img.source = img_path
        self.ids.image_path_input.text = img_path
        
class VGG16_Quality_InsuranceApp(App) : # 창 이름
    def build(self) :
    
        return ClassificationApp()

if __name__ == '__main__' :
    VGG16_Quality_InsuranceApp().run()