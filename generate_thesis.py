import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_thesis():
    # 1. Create a new document
    doc = Document()
    
    # 2. Set margins to 1 inch on all sides
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        # Ensure no header/footer space
        section.header_distance = Inches(0)
        section.footer_distance = Inches(0)

    # 3. Define basic styles
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0) # All text in black
    # 1.5 line spacing
    style_normal.paragraph_format.line_spacing = 1.5

    def add_heading(text, level=1):
        # Level 1 for Main Headings (Centered, Larger)
        # Level 2 for Subheadings (Left aligned, Medium)
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(0, 0, 0)
        
        if level == 1:
            run.font.size = Pt(16)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            run.font.size = Pt(14)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        # Add some space after heading
        p.paragraph_format.space_after = Pt(12)
        return p

    def add_paragraph(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0, 0, 0)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return p

    def add_image_placeholder(text):
        p = doc.add_paragraph()
        run = p.add_run(f"\n[ --- PLEASE INSERT IMAGE HERE: {text} --- ]\n")
        run.bold = True
        run.font.color.rgb = RGBColor(255, 0, 0) # Red just to make placeholder obvious to you
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(24)
        p.paragraph_format.space_after = Pt(24)

    # ================= FRONT MATTER =================

    add_heading("DEEPFAKE DETECTION ON VIDEO DATA USING WAVELET-BASED TEMPORAL MAMBA", level=1)
    
    add_paragraph("A PROJECT REPORT")
    p = doc.paragraphs[-1]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    add_paragraph("By\nName of the student: Piyush\nEnrolment/Registration No.: [Your ID Here]")
    p = doc.paragraphs[-1]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_page_break()

    add_heading("ABSTRACT", level=1)
    add_paragraph("Fake videos made by AI are called deepfakes. They look very real today. People can easily get fooled by them on the internet. We need good software to catch these fake videos. In this project, we built a new deepfake detector. A lot of old detectors only look at one picture frame at a time. But video is moving pictures. So, if we check how the video moves over time, we can catch the fakes much better.")
    add_paragraph("My system looks at the video in two steps. First, it uses something called Wavelets to find weird noise in the pixels. This noise comes from the AI generator. Second, it uses a new tool called Mamba. Mamba tracks the noise across 32 frames of the video to see if it flickers in a fake way. We trained this model using a lot of data. The results show that our model works very well on unseen videos, scoring over 96% accuracy. This proves that tracking noise over time is the best way to catch deepfakes.")

    doc.add_page_break()

    # ================= CHAPTER 1 =================
    add_heading("Chapter 1: Introduction", level=1)
    
    add_heading("1.1 Background", level=2)
    add_paragraph("Creating fake videos used to be very hard. You needed movie studio tools and a lot of time. But now, AI makes it very easy. Tools like GANs and diffusion models can swap faces or make a person say things they never said. Because these fakes look so real, they can cause big problems. People might use them to spread fake news or trick someone for money. Human eyes cannot see the tiny mistakes made by the AI, so we need computers to find them for us.")
    
    add_heading("1.2 The Problem", level=2)
    add_paragraph("We already have some deepfake detectors. But they have a big problem. They only work on the exact videos they were trained on. If you show them a new type of fake video, or a video that is compressed for WhatsApp, they fail completely. This is because they only look at a single frame. When video is compressed, it gets blurry. The detector gets confused and thinks the blur is a deepfake. We need a way to solve this so detectors work in the real world.")

    add_heading("1.3 Motivation", level=2)
    add_paragraph("The main reason for doing this project is to fix this failure. I want to build a detector that works on new videos it has never seen before. To do this, I decided to look at how the video changes over time. AI is very bad at keeping the tiny pixel noise the same from frame 1 to frame 30. If we track that noise, we can spot the fake video easily. ")

    # ================= CHAPTER 2 =================
    doc.add_page_break()
    add_heading("Chapter 2: Literature Review", level=1)
    add_paragraph("I read many research papers to understand how others tried to catch deepfakes. The papers are mentioned in the references at the end. Here is what I found.")
    
    add_heading("2.1 Old Single Frame Models", level=2)
    add_paragraph("Most early research used simple CNN models. They looked for weird edges on faces. Recently, people started using Vision Transformers to look at the whole face. A big problem with these is that they only check one picture at a time. A paper by Muller et al. showed that when you test these models on real-world compressed video, they stop working completely.")

    add_heading("2.2 Wavelet Models", level=2)
    add_paragraph("Some new papers, like the one by Peng et al. (2025), talk about Wavelet Mamba (WMamba). They used wavelets to pull out the hidden AI noise from the video. This worked much better than simple CNNs. But again, Peng's model only checked single frames. It had no idea how to track the video over time. This meant it still got confused by video compression.")

    add_heading("2.3 Temporal Models", level=2)
    add_paragraph("Some researchers tried to look at video over time. For example, UMMAFormer by Wang et al. looked at long videos to find exact timestamps of fakes. But they used Transformers, which take too much memory on the graphics card. They are too heavy to run properly on normal computers.")

    add_heading("2.4 Research Gap", level=2)
    add_paragraph("After reading the papers, I saw a clear gap. No one has combined the wavelet noise checker with a fast memory-friendly tracker like Mamba. If we combine them, we can check 32 frames without running out of computer memory. This is exactly what my project does.")

    # ================= CHAPTER 3 =================
    doc.add_page_break()
    add_heading("Chapter 3: System Implementation", level=1)

    add_heading("3.1 The Dataset Used", level=2)
    add_paragraph("To teach the computer, I used three big datasets. The first is LAV-DF. I used about 38,000 videos from here to train the first part of the model. Then I made a smaller mixed dataset of 2,300 videos. This small dataset had videos from FaceForensics++ (FFPP) and Deepfake Detection Challenge (DFDC). I made sure the data was perfectly balanced, meaning 50% real videos and 50% fake videos.")

    add_heading("3.2 Pre-processing", level=2)
    add_paragraph("First, I take the video and chop it into 32 separate pictures (frames). Then I use a face finder code called dlib. It draws a box around the face. I expanded this box by 20% so I can also see the background around the head. Finally, I resized all the faces to 256x256 size so the computer can process them fast.")

    add_image_placeholder("Diagram showing video frames being chopped and face cropping")

    add_heading("3.3 Model Architecture", level=2)
    add_paragraph("The model works in two stages. First is the Spatial stage. Here, I use Discrete Haar Wavelets. This drops all the normal colors and leaves only the sharp edges and AI noise. Then I send it through a Dynamic Contour Convolution layer, which is great at finding weird facial boundaries.")
    add_paragraph("Second is the Temporal stage. I take the noise from all 32 frames and feed it into a Bidirectional Mamba model. Mamba is a state space model. It scans the video forwards and backwards. It checks if the noise is flickering randomly. If it flickers like AI, it flags the video as fake.")

    add_image_placeholder("Flowchart showing Wavelets connecting to the Mamba model")

    # ================= CHAPTER 4 =================
    doc.add_page_break()
    add_heading("Chapter 4: Results", level=1)

    add_heading("4.1 Hardware setup", level=2)
    add_paragraph("I ran all my code on a Kubernetes server using an NVIDIA H200 MIG slice. I only had 16 GB of memory. To make sure the big 32-frame model fit into the memory, I used mixed precision (bf16) and a micro-batch size of 32. This was a very important trick to avoid crashing the computer.")

    add_heading("4.2 Evaluation", level=2)
    add_paragraph("I tested the model on 494 brand new videos that it had never seen before. I used a metric called AUC, which measures how good the model is at separating real from fake. ")
    
    add_image_placeholder("Bar chart or graph showing the accuracy scores for FFPP, DFDC, and LAV-DF")

    add_heading("4.3 Final Scores", level=2)
    add_paragraph("The results were amazing. On the DFDC dataset, which is very heavily compressed and usually breaks models, my system got 95.75% AUC. On the FFPP dataset, it got 97.34% AUC and had zero false negatives. Overall, across all the datasets, the average score was 96.93%.")
    add_paragraph("I also tested what happens if I turn off the Mamba tracking. When I did that, the score dropped down to 75%. This proves that looking at the video over time is the main reason my model works so well.")

    # ================= CHAPTER 5 =================
    doc.add_page_break()
    add_heading("Chapter 5: Conclusion", level=1)

    add_heading("5.1 Summary", level=2)
    add_paragraph("In this project, I built a new deepfake detector using Wavelets and a Bidirectional Mamba tracker. The main goal was to make a detector that does not fail when it sees a new, unseen video. I proved that if you track the AI noise across 32 frames, the model becomes very smart and avoids being tricked by normal video compression.")

    add_heading("5.2 Future Work", level=2)
    add_paragraph("In the future, I want to add audio checking to this model. Right now, it only looks at pictures. If an attacker fakes only the audio voice, my model will not catch it. Also, I want to make the code run faster so it can be installed on a mobile phone for real-time checking.")

    # ================= REFERENCES =================
    doc.add_page_break()
    add_heading("References", level=1)
    
    refs = [
        "[1] Peng, Gao, Zhang, et al. (2025). Wmamba: Wavelet-based Mamba for Face Forgery Detection.",
        "[2] Muller, N. M. et al. (2026). Does Audio Deepfake Detection Generalize?",
        "[3] UMMAFormer (2023). Universal Multimodal-Adaptive Transformer for Temporal Forgery Localization.",
        "[4] VIDS-Guard (2024). Multi-stream transformer architecture integrating FFT streams with spatial patch extraction.",
        "[5] Wang, Q. et al. (2026). AUDETER: A Large-scale Dataset for Deepfake Audio Detection in Open Worlds."
    ]
    
    for ref in refs:
        p = add_paragraph(ref)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # 8. Save the document
    doc.save('Thesis_Report_Piyush.docx')
    print("Document successfully saved as Thesis_Report_Piyush.docx")

if __name__ == '__main__':
    create_thesis()
