import argparse
import os
import numpy as np
import tkinter as tkr
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

import torch
from torch import nn
from PIL import Image, ImageTk, ImageChops, ImageEnhance, ImageOps


class FakeDetector(nn.Module):
    def __init__(self, input_channels, dropout=0.5):
        super(FakeDetector, self).__init__()
        
        self.conv11     = nn.Conv2d(in_channels=input_channels, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv11_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn11       = nn.BatchNorm2d(10)
        self.conv12     = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv12_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn12       = nn.BatchNorm2d(10)
        self.conv13     = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv13_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn13       = nn.BatchNorm2d(10)
        
        self.conv21     = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv21_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn21       = nn.BatchNorm2d(10)
        self.conv22     = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv22_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn22       = nn.BatchNorm2d(10)
        self.conv23     = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv23_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn23       = nn.BatchNorm2d(10)
        
        self.conv31     = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv31_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn31       = nn.BatchNorm2d(10)
        self.conv32     = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv32_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn32       = nn.BatchNorm2d(10)
        self.conv33     = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv33_1x1 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.bn33       = nn.BatchNorm2d(10)
        
        self.fc1        = nn.Linear(in_features=10*25*25, out_features=100)
        self.fc2        = nn.Linear(in_features=100, out_features=100)
        self.fc3        = nn.Linear(in_features=100, out_features=2)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(dropout, True)

    def forward(self, inputs):
        output10 = inputs.clone()
        
        output = self.conv11(inputs)
        output = self.leaky_relu(output)
        output = self.conv11_1x1(output)
        output11 = self.bn11(output)
#         output11 = output
        output = output10 + output11
        
        output = self.conv12(output)
        output = self.leaky_relu(output)
        output = self.conv12_1x1(output)
        output12 = self.bn12(output)
#         output12 = output
        output = output11 + output12
        
        output = self.conv13(output)
        output = self.leaky_relu(output)
        output = self.conv13_1x1(output)
        output = self.bn13(output)
        output += output12
        
        #####################################
        output = self.max_pool2d(output)
        output20 = output.clone()
        
        output = self.conv21(output)
        output = self.leaky_relu(output)
        output = self.conv21_1x1(output)
        output21 = self.bn21(output)
#         output21 = output
        output = output20 + output21
        
        output = self.conv22(output)
        output = self.leaky_relu(output)
        output = self.conv22_1x1(output)
        output22 = self.bn22(output)
#         output22 = output
        output = output21 + output22
        
        output = self.conv23(output)
        output = self.leaky_relu(output)
        output = self.conv23_1x1(output)
        output = self.bn23(output)
        output += output22
        
        #####################################
        output = self.max_pool2d(output)
        output30 = output.clone()
        
        output = self.conv31(output)
        output = self.leaky_relu(output)
        output = self.conv31_1x1(output)
        output31 = self.bn31(output)
#         output31 = output
        output = output30 + output31
        
        output = self.conv32(output)
        output = self.leaky_relu(output)
        output = self.conv32_1x1(output)
        output32 = self.bn32(output)
#         output32 = output
        output = output31 + output32
        
        output = self.conv33(output)
        output = self.leaky_relu(output)
        output = self.conv33_1x1(output)
        output = self.bn33(output)
        output += output32
        
        #####################################
        output = output.view(output.size(0), -1) # flatten
        
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.leaky_relu(output)
        
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.leaky_relu(output)
        
        output = self.dropout(output)
        output = self.fc3(output)
        
        return output

def get_test_image(filename):
    img = Image.open(filename).convert('RGB')
    imgs = []
    cvals = [90, 85, 80, 75, 95, 70, 65, 60, 55, 50]
    
    for q in cvals:
        tempname = filename+'temp'+str(q)
        img.save(tempname, 'JPEG', quality=q)
        temp_img = Image.open(tempname)
        ela_img = ImageChops.difference(img, temp_img)
        os.remove(tempname)
        
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0/max_diff
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
        
        ela_img = ImageOps.fit(ela_img, (100, 100), Image.ANTIALIAS, 0, (0.5, 0.5)).convert('L')
        imgs.append(np.array(ela_img))

    return np.asarray([imgs])

def init():
    use_gpu = torch.cuda.is_available()
    model = FakeDetector(input_channels=10, dropout=0.5)
    model.cpu()
    if os.path.isfile('./model_cpu.pt'):
        model.load_state_dict(torch.load('./model_cpu.pt'))
    else:
        print("Error, saved model not found.")
    model = model.cuda() if use_gpu else model.cpu()
    return model, use_gpu

def test(model, filename, use_gpu=False):
    softmax = torch.nn.Softmax(dim=1)
    test_img = get_test_image(filename)
    inputs = torch.from_numpy(test_img).float()
    inputs = inputs.cuda() if use_gpu else inputs

    model.eval()
    output = model(inputs)
    print(output)

    is_real = np.argmax(output.detach(), axis=1)[0].item()
    confidence_level = softmax(output).detach()[0][is_real].item()
    
    print('is_real', is_real)
    print('confidence_level', confidence_level)
    
    return is_real, confidence_level

class Application(ttk.Frame):
    def __init__(self, master=None):
        if master is not None:
            master.title("Computer Vision: Fake Image Detection")
        super().__init__(master)
        self.pack(fill="both", expand=True)
        
        self.model, self.use_gpu = init()
        
        self.photoimage = None
        self.ela_image = None
        
        # Image frame
        image_frame = ttk.Frame(self, height=500, width=500)
        
        self.image_button = ttk.Button(image_frame, text="Select image", command=self.select_image)
        self.image_button.pack(fill="both", expand=True, side = "left")
        # Ela Widget
        self.ela_label = ttk.Button(image_frame, text="To begin, press the \n button on the left \n to select an image")
        self.ela_label.pack(fill="both", expand=True,side = "right")
        
        image_frame.pack(expand=True, fill='both')
        
        self.exposition = ttk.Label(self,text = "The panel on the left displays the image and the panel on the right displays the Error Level Analysis of the image.")
        self.exposition.pack()
        self.answerstr = tkr.StringVar()
        self.answerstr.set("Answer will be displayed here")
        self.answer_box = tkr.Message(self, textvar=self.answerstr)
        self.answer_box.pack(fill="x")
        self.answer_box.bind("<Configure>", lambda event: self.answer_box.configure(width=event.width-10))
        
        self.master.minsize(500,500)
    
    def select_image(self, event=None):
        image_path = filedialog.askopenfilename(filetypes = (("all files","*.*"),))
        with open(image_path, 'rb') as f:
            try:
                img = Image.open(f)
            except OSError as e:
                messagebox.showinfo("Error", "Cannot open image: {}".format(e))
                return
            img = img.convert('RGB')
        
        self.photoimage = ImageTk.PhotoImage(img)
        self.image_button.configure(image=self.photoimage)
        
        # Dsplay the ela image here 
        self.ela_image = self.convertELA(image_path)
        self.ela_label.configure(image=self.ela_image)
        
        self.answerstr.set("Generating results...")
        is_real, confidence_level = test(self.model, image_path, use_gpu=self.use_gpu)
        is_auth = "authentic" if is_real else "fake"
        ans = "\nAccording to our analysis, the image above is {:s}!\nConfidence level: {:3f}%\n".format(is_auth, confidence_level*100)
        self.answerstr.set(ans)
        
    def convertELA(self, path):
        quality = 75
        TMP_EXT = ".tmp_ela.jpg"
        ELA_EXT = ".ela.png"
        """
        Generates an ELA image on save_dir.
        Params:
            fname:      filename w/out path
            orig_dir:   origin path
            save_dir:   save path
        """
        fname = os.path.basename(path)
        save_dir = ".\Temp"
        basename, ext = os.path.splitext(fname)
        
        tmp_fname = os.path.join(save_dir, basename + TMP_EXT)
        ela_fname = os.path.join(save_dir, basename + ELA_EXT)
    
        im = Image.open(path)
        # Save a temporary copy of an image as a .jpg file
        im.save(tmp_fname, 'JPEG', quality=quality)
        
        # Open the temp-file
        tmp_fname_im = Image.open(tmp_fname)
        # EExtract the difference between the temporarily saved file and the current image 
        ela_im = ImageChops.difference(im, tmp_fname_im)
        
        # Gets the the minimum and maximum pixel values for each band in the image [R,G,B]
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        scale = 255.0/max_diff
        
        # A factor of 1 gives the original image; hence we accentuate the compression errors
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale+5)
        os.remove(tmp_fname)
        return ImageTk.PhotoImage(ela_im)


def main():
    app = Application(master=tkr.Tk())
    app.mainloop()



if __name__ == "__main__":
    main()