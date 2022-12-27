


from glob import glob
from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt
RAF = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\output"
FER = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\FER2013\train"
RAFER  = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAFER"

if __name__ == "__main__":
    
    for raf in glob(RAF + "\*\*"):
        rafpth = Path(raf)
        
        print("***************************** ", rafpth.parent.name)
        

        # **************************** angry
        if(rafpth.parent.name == "angry"):
            if (os.path.exists(RAFER + r"\\angry") == False):
                os.mkdir(RAFER + r"\\angry")

            image = Image.open(raf)
            image.save(RAFER+ r"\\angry\\" + raf.split("\\")[-1])


        # **************************** disgust
        if(rafpth.parent.name == "disgust"):
            if (os.path.exists(RAFER + r"\\disgust") == False):
                os.mkdir(RAFER + r"\\disgust")

            image = Image.open(raf)
            image.save(RAFER+ r"\\disgust\\" + raf.split("\\")[-1])

        
        # **************************** fear
        if(rafpth.parent.name == "fear"):
            if (os.path.exists(RAFER + r"\\fear") == False):
                os.mkdir(RAFER + r"\\fear")

            image = Image.open(raf)
            image.save(RAFER+ r"\\fear\\" + raf.split("\\")[-1])
        
        # **************************** happy
        if(rafpth.parent.name == "happy"):
            if (os.path.exists(RAFER + r"\\happy") == False):
                os.mkdir(RAFER + r"\\happy")

            image = Image.open(raf)
            image.save(RAFER+ r"\\happy\\" + raf.split("\\")[-1])
        
        # **************************** neutral
        if(rafpth.parent.name == "neutral"):
            if (os.path.exists(RAFER + r"\\neutral") == False):
                os.mkdir(RAFER + r"\\neutral")

            image = Image.open(raf)
            image.save(RAFER+ r"\\neutral\\" + raf.split("\\")[-1])

        
        # **************************** sad
        if(rafpth.parent.name == "sad"):
            if (os.path.exists(RAFER + r"\\sad") == False):
                os.mkdir(RAFER + r"\\sad")

            image = Image.open(raf)
            image.save(RAFER+ r"\\sad\\" + raf.split("\\")[-1])


        # **************************** surprise
        if(rafpth.parent.name == "surprise"):
            if (os.path.exists(RAFER + r"\\surprise") == False):
                os.mkdir(RAFER + r"\\surprise")

            image = Image.open(raf)
            image.save(RAFER+ r"\\surprise\\" + raf.split("\\")[-1])
    


    for fer in  glob(FER + "\*\*"):
        ferpth = Path(fer)
        print("***************************** ", ferpth.parent.name)

        # **************************** angry
        if(ferpth.parent.name == "angry"):
            if (os.path.exists(RAFER + r"\\angry") == False):
                os.mkdir(RAFER + r"\\angry")

            image = Image.open(fer)
            image.save(RAFER+ r"\\angry\\" + fer.split("\\")[-1][:-3] + "png")

        # **************************** disgust
        if(ferpth.parent.name == "disgust"):
            if (os.path.exists(RAFER + r"\\disgust") == False):
                os.mkdir(RAFER + r"\\disgust")

            image = Image.open(fer)
            image.save(RAFER+ r"\\disgust\\" + fer.split("\\")[-1][:-3] + "png")
        # **************************** fear
        if(ferpth.parent.name == "fear"):
            if (os.path.exists(RAFER + r"\\fear") == False):
                os.mkdir(RAFER + r"\\fear")

            image = Image.open(fer)
            image.save(RAFER+ r"\\fear\\" + fer.split("\\")[-1][:-3] + "png")

        # **************************** happy
        if(ferpth.parent.name == "happy"):
            if (os.path.exists(RAFER + r"\\happy") == False):
                os.mkdir(RAFER + r"\\happy")

            image = Image.open(fer)
            image.save(RAFER+ r"\\happy\\" + fer.split("\\")[-1][:-3] + "png")

        # **************************** neutral
        if(ferpth.parent.name == "neutral"):
            if (os.path.exists(RAFER + r"\\neutral") == False):
                os.mkdir(RAFER + r"\\neutral")

            image = Image.open(fer)
            image.save(RAFER+ r"\\neutral\\" + fer.split("\\")[-1][:-3] + "png")

        # **************************** sad
        if(ferpth.parent.name == "sad"):
            if (os.path.exists(RAFER + r"\\sad") == False):
                os.mkdir(RAFER + r"\\sad")

            image = Image.open(fer)
            image.save(RAFER+ r"\\sad\\" + fer.split("\\")[-1][:-3] + "png")

        # **************************** surprise
        if(ferpth.parent.name == "surprise"):
            if (os.path.exists(RAFER + r"\\surprise") == False):
                os.mkdir(RAFER + r"\\surprise")

            image = Image.open(fer)
            image.save(RAFER+ r"\\surprise\\" + fer.split("\\")[-1][:-3] + "png")

        

        

        

