"""
Internship Report Generator for Nationality Detection System
Creates a professional PDF report documenting the internship project.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors

def create_report():
    doc = SimpleDocTemplate(
        "Internship_Report_Task4_Nationality_Detection.pdf",
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=6,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.darkblue
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=16
    )
    
    story = []
    
    # Title Page
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Internship Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Task 4: Nationality Detection System", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("A Multi-Task Computer Vision Application for Human Attribute Analysis", subtitle_style))
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("<b>Prepared by:</b> Tarrush Saxena", subtitle_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Completed:</b> January 2026", subtitle_style))
    story.append(Spacer(1, 2*inch))
    
    # Introduction
    story.append(Paragraph("1. Introduction", heading_style))
    intro_text = """
    This report documents my work on Task 4 of the internship program, which involved developing 
    a Nationality Detection System. The project challenged me to build a complete computer vision 
    application that can detect and analyze multiple human attributes from a live camera feed. 
    What started as a straightforward classification task quickly evolved into something much more 
    interesting—a multi-task learning system that detects nationality, emotion, age, and even 
    clothing color, all in real-time. I found myself diving deep into transfer learning, neural 
    network architectures, and designing a user interface that could display all this information 
    in an intuitive way. The experience really pushed me to integrate everything I had learned 
    about deep learning, image processing, and GUI development into one cohesive project.
    """
    story.append(Paragraph(intro_text, body_style))
    
    # Background
    story.append(Paragraph("2. Background", heading_style))
    background_text = """
    The idea behind this project comes from the growing need in various industries to understand 
    demographic and emotional patterns. Think about retail analytics where stores want to know 
    who their customers are, or hospitality services that could benefit from understanding guest 
    demographics. Even security systems are starting to incorporate these kinds of technologies.
    
    The core problem I tackled was this: given a video stream from a webcam, can we reliably 
    identify a person's nationality or ethnicity, determine their emotional state, estimate their 
    age, and pick up on what color clothes they're wearing? Each of these is a complex problem 
    on its own, but the real challenge was making them work together smoothly and fast enough 
    for real-time processing. I used OpenCV for handling the video feed and face detection, 
    TensorFlow and Keras for the deep learning models, and scikit-learn for the color clustering 
    algorithm. For the user interface, I went with Tkinter because it's reliable and doesn't 
    add unnecessary complexity.
    """
    story.append(Paragraph(background_text, body_style))
    
    # Learning Objectives
    story.append(Paragraph("3. Learning Objectives", heading_style))
    objectives_text = """
    Going into this project, I set myself some clear goals about what I wanted to learn and 
    accomplish. Here's what I was aiming for:
    """
    story.append(Paragraph(objectives_text, body_style))
    
    objectives = [
        "Get hands-on experience with transfer learning using pre-trained models like MobileNetV2",
        "Build and train CNNs for different tasks—classification for nationality and emotion, regression for age",
        "Figure out how to do real-time face detection and tracking with OpenCV",
        "Learn K-Means clustering for practical image analysis like color extraction",
        "Put together a complete end-to-end ML pipeline from data prep to deployment",
        "Design a graphical interface that updates smoothly with real-time predictions",
        "Understand how to organize a multi-model system with proper configuration management"
    ]
    
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", body_style))
    
    # Activities and Tasks
    story.append(Paragraph("4. Activities and Tasks", heading_style))
    activities_text = """
    The project work broke down into several distinct phases, each building on the previous one:
    
    <b>Dataset Preparation:</b> I spent quite a bit of time gathering and organizing training data. 
    For nationality detection, I worked with the FairFace dataset, which has reasonably balanced 
    representation across different ethnic groups. For emotion recognition, I used FER-2013, which 
    is a classic dataset with seven emotion categories. Age prediction training used the UTKFace 
    dataset. The data setup script I wrote handles all the downloading, extraction, and folder 
    organization automatically—it saves a lot of time when you need to set up the project on a 
    new machine.
    
    <b>Model Architecture Design:</b> For the nationality model, I used MobileNetV2 as a base and 
    added custom classification layers on top. MobileNetV2 is great because it's efficient enough 
    for real-time inference but still powerful enough to learn complex patterns. The emotion and 
    age models use more traditional CNN architectures since they work with smaller input images 
    and don't need the same level of feature extraction capability.
    
    <b>Training Pipeline:</b> I wrote separate training scripts for each model, all with argument 
    parsing so you can easily adjust hyperparameters like epochs, batch size, and learning rate 
    from the command line. Each script includes data augmentation, early stopping, and model 
    checkpointing to save the best weights. The training process outputs progress updates so you 
    can monitor how things are going.
    
    <b>Color Extraction Module:</b> This was an interesting sub-project. Rather than using another 
    neural network, I went with a K-Means clustering approach to find the dominant color in the 
    torso region (below the detected face). The algorithm clusters pixel colors and returns the 
    most common one, which I then map to a human-readable color name using Euclidean distance 
    to a predefined color palette.
    
    <b>Engine Integration:</b> The NationalityEngine class ties everything together. It loads all 
    the models, handles preprocessing for each one, and provides a clean interface for the GUI 
    to call. The engine also implements a branching logic system—depending on the detected 
    nationality, different attributes might be computed or displayed differently.
    
    <b>GUI Development:</b> I built the interface using Tkinter with a clean, professional look. 
    The main window shows the live camera feed with bounding boxes and labels overlaid on detected 
    faces. A side panel displays execution logs so you can see what the system is doing. The models 
    load in a background thread so the UI stays responsive during startup.
    """
    story.append(Paragraph(activities_text, body_style))
    
    # Skills and Competencies
    story.append(Paragraph("5. Skills and Competencies", heading_style))
    skills_text = """
    Working on this project really leveled up my skills in several key areas:
    
    <b>Deep Learning:</b> I got much more comfortable with TensorFlow and Keras, especially when 
    it comes to transfer learning. Understanding how to freeze base layers, add custom heads, 
    and fine-tune models was invaluable. I also learned a lot about handling different model 
    types—classification vs regression—and how the loss functions and output layers differ.
    
    <b>Computer Vision:</b> OpenCV became second nature by the end of this project. Haar cascades 
    for face detection, image preprocessing, color space conversions, drawing annotations on 
    frames—all of these are now tools I can reach for confidently.
    
    <b>Python Development:</b> The project reinforced good practices like modular code organization, 
    configuration management, argument parsing for scripts, and threading for responsive UIs. 
    I also got better at writing code that fails gracefully—the system handles missing models 
    or failed predictions without crashing.
    
    <b>Problem Solving:</b> There were so many little puzzles to figure out. How do you estimate 
    where someone's torso is based on their face location? How do you make model inference fast 
    enough for real-time video? How do you display multiple attributes on screen without making 
    it look cluttered? Each of these required creative thinking and experimentation.
    """
    story.append(Paragraph(skills_text, body_style))
    
    # Feedback and Evidence
    story.append(Paragraph("6. Feedback and Evidence", heading_style))
    evidence_text = """
    The finished system works quite well in practice. Here's what I observed during testing:
    
    The nationality detection runs at a comfortable frame rate on my laptop—fast enough that 
    there's no noticeable lag when moving in front of the camera. The emotion recognition is 
    surprisingly responsive, picking up changes in facial expressions almost instantly. Age 
    predictions are in the right ballpark, though like most age estimation systems, they 
    sometimes miss by a few years.
    
    The color detection feature is probably the most fun to watch. When I change shirts or 
    stand in front of different backgrounds, the system correctly identifies the dominant 
    color most of the time. It occasionally gets confused with very complex patterns or 
    similar shades, but overall performs well.
    
    The code is well-documented and organized into clear modules. The configuration file 
    makes it easy to adjust model paths, color detection parameters, and branching logic 
    without touching the core code. The project includes unit tests and a comprehensive 
    README with installation and usage instructions.
    """
    story.append(Paragraph(evidence_text, body_style))
    
    # Challenges and Solutions
    story.append(Paragraph("7. Challenges and Solutions", heading_style))
    challenges_text = """
    Like any substantial project, this one came with its share of headaches:
    
    <b>Model Loading Errors:</b> Initially, loading saved Keras models would fail with cryptic 
    errors about missing metrics or custom objects. The fix turned out to be using compile=False 
    when loading, then recompiling if needed. This taught me a lot about how Keras serializes 
    model configurations.
    
    <b>Real-Time Performance:</b> My first implementation was way too slow—predictions were 
    lagging behind the video by several seconds. I solved this by using smaller input sizes 
    where possible, reducing the number of K-Means iterations for color detection, and being 
    smarter about when to run predictions (not every single frame needs full analysis).
    
    <b>Dataset Quality:</b> Some of the datasets I used have known biases and quality issues. 
    For example, lighting conditions in training images don't always match real webcam conditions. 
    Data augmentation during training helped somewhat, but this remains an area for improvement.
    
    <b>Threading and UI Responsiveness:</b> The GUI would freeze while models were loading at 
    startup. I fixed this by loading models in a background thread and updating the UI when 
    loading completes. It seems simple now, but getting the threading right without race 
    conditions took some careful thought.
    
    <b>Torso Region Estimation:</b> Figuring out where to look for clothing color wasn't 
    straightforward. The face bounding box only tells you where the face is, not where the 
    body is. I developed a heuristic that extends below and slightly to the sides of the face, 
    which works well for typical webcam angles.
    """
    story.append(Paragraph(challenges_text, body_style))
    
    # Outcomes and Impact
    story.append(Paragraph("8. Outcomes and Impact", heading_style))
    outcomes_text = """
    By the end of this task, I had built a fully functional multi-attribute detection system 
    that processes live video in real-time. Here's what the final product accomplishes:
    
    <b>Technical Deliverables:</b>
    • Three trained deep learning models (nationality, emotion, age) with saved weights
    • A modular codebase with separate modules for GUI, engine, and color extraction
    • Training scripts with configurable parameters and automatic model checkpointing
    • Data preparation utilities that handle dataset download and organization
    • A complete configuration system for customizing model paths and detection parameters
    
    <b>Practical Capabilities:</b>
    • Detects faces in real-time video and draws color-coded bounding boxes
    • Predicts nationality/ethnicity with confidence scores displayed on screen
    • Recognizes seven different emotions and updates predictions live
    • Estimates age using a regression model trained on face images
    • Identifies dominant clothing color using unsupervised clustering
    
    <b>Learning Outcomes:</b>
    This project gave me end-to-end experience with a real ML application. I now understand 
    what it takes to go from raw datasets to a working product that regular users can interact 
    with. The skills I developed—transfer learning, real-time processing, multi-model systems, 
    UI development—are directly applicable to other computer vision projects.
    """
    story.append(Paragraph(outcomes_text, body_style))
    
    # Conclusion
    story.append(Paragraph("9. Conclusion", heading_style))
    conclusion_text = """
    Task 4 was probably the most comprehensive project I worked on during this internship. 
    Building the Nationality Detection System pushed me to integrate multiple machine learning 
    models, work with real-time video processing, and create a user-friendly interface—all 
    in one cohesive application.
    
    The biggest takeaway for me is understanding how different components of an ML system 
    need to work together. It's not enough to train a good model; you also need efficient 
    inference, clean code architecture, error handling, and a way for users to actually 
    interact with your work. This project covered all of those aspects.
    
    Looking forward, there are definitely ways this system could be improved. Better training 
    data, more sophisticated face detection algorithms, and GPU acceleration for inference 
    would all help. But as a learning experience and a demonstration of multi-task computer 
    vision, I'm proud of what I built here.
    
    The complete source code, trained models, and documentation are organized in a clean 
    project structure and ready for future development or deployment.
    """
    story.append(Paragraph(conclusion_text, body_style))
    
    # Build PDF
    doc.build(story)
    print("PDF report generated successfully!")
    print("File: Internship_Report_Task4_Nationality_Detection.pdf")

if __name__ == "__main__":
    create_report()
