import cv2
import time
import torch
from PIL import Image
import numpy as np
import cam_models
import draw_bar
import dlib
import face_utils


from torchvision import transforms
# Resnet wants images represented in a certain format;
# when called, this function will take care of that.
transform = transforms.Compose([               #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
        mean=[0.485, 0.456, 0.406],            #[6]
        std=[0.229, 0.224, 0.225]              #[7]
    )])


labels_list = []
with open("map_clsloc.txt", "r") as labels_file:
    for line in labels_file:
        _, _, label = line.split(sep=" ")
        label = label.strip()
        labels_list.append(label)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

## Modified resnet model
res_pretrained = cam_models.MyResNetModel()
res_pretrained = res_pretrained.to(device)

## Our model - unavailable on github, missing custom trained model
#res_rnn_model = cam_models.resnet_only(3)
#res_rnn_model.load_state_dict(torch.load('/mnt/Data/saved-models/resnet-rmsprop-mse-short#0.model'))
#res_rnn_model = res_rnn_model.to(device)

def model_pre_and_run(three_frames):

    # Transform our three frames
    transformed_frame_list = [Image.fromarray(frame) for frame in three_frames]
    transformed_frame_list = [transform(frame) for frame in transformed_frame_list]
    transformed_frame_list = [frame.view((3, 224, 224)) for frame in transformed_frame_list]

    # Stack them, so the tensor dimensions will be frame by x by y
    stacked_tensor = torch.stack(transformed_frame_list)
    stacked_tensor = stacked_tensor.to(device)

    # Get our 3 by 2048 resnet output
    resnet_result_vector = res_pretrained(stacked_tensor)
    resnet_result_vector = resnet_result_vector.view(3, 2048).to(device)

    #twenty_stack = [resnet_result_vector for i in range(20)]
    #twenty_stack = torch.stack(transformed_frame_list)
    #twenty_stack = twenty_stack.to(device)

    # Run through rnn
    res_pre_out = res_pretrained(resnet_result_vector)


    mem_score = res_rnn_out.item()

    # The output mem score doesn't vary that much... makes me think that the resnet
    # output isn't varying that much-- remember trilobyte/iguana?
    return mem_score

def main():

    draw_font = cv2.FONT_HERSHEY_SIMPLEX

    # Various methods for face detection, uncomment only 1!
    #face_cascade = cv2.CascadeClassifier('/home/alex/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    #face_hog = dlib.get_frontal_face_detector()
    face_cnn = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    face_landmarks = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cv2.namedWindow("Memory Cam")
    cv2.resizeWindow("Memory Cam", 800, 600)
    # Initialize our webcam/video
    video_capture = cv2.VideoCapture(-1)
    video_capture.set(3, 320)
    video_capture.set(4, 240)

    if video_capture.isOpened():
        valid_read, frame = video_capture.read()

    else:
        valid_read = False

    #f_width, f_height = frame.size

    time_run_start = time.time()
    num_frames = 0

    # A sort of cache for frames; once this
    # reaches a size of three, we pass this to
    # model_pre_and_run to be transformed and used as
    # model input.
    three_frames_list = []
    score = 0.0

    while valid_read:
        time_start = time.time()

        if len(three_frames_list) == 3:
            # Run rnn model over 3 frames
            score = model_pre_and_run(three_frames_list)
            three_frames_list = []

        score_fmt = "{0:.2f}".format(score)
        print_string = "Score: " + score_fmt

        # Draw a text representation of the score over the frame
        cv2.putText(frame, print_string, (10, 20), draw_font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw a meter representation of the score over the frame
        draw_bar.mem_meter(score, frame)

        # Store a grayscale version of the frame for the
        # face cascade and landmark detector to use. The frame
        # will still be displayed on the screen in color.
        # Uncomment only one 'faces' face detection method. 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #faces = face_hog(gray, 1)
        faces = face_cnn(frame, 1)
        #faces = []

        # We're iterating over each detected face
        for face in faces:
            # For CNN
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y

            # For HOG
            #x = face.left()
            #y = face.top()
            #w = face.right() - x
            #h = face.bottom() - y

            # Draw the detected rectangle bounds over the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # dlib wants a particular rectangle object for use in face_landmarks
            face_rect = dlib.rectangle(x, y, x + w, y + h)
            # Detect landmarks
            marks = face_landmarks(gray, face_rect)
            # Convert landmark coordinates to tuples
            mark_pt_coords = face_utils.shape_to_np(marks)

            for (x, y) in mark_pt_coords:
                # Draw points at coordinates on the frame
                cv2.circle(frame, (x, y), 1, (225, 225, 225), -1)

        # Finally, display the frame on the screen.

        big_frame = cv2.resize(frame, (640, 480), cv2.INTER_CUBIC)
        cv2.imshow("Memory Cam", big_frame)
        valid_read, frame = video_capture.read()

        if valid_read:
            # Capture the frame
            three_frames_list.append(frame)

        num_frames += 1

        #frame_queue.put(frame)
        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC key
            break

    time_run_end = time.time()

    time_run = time_run_end - time_run_start

    print("FPS:", str(float(num_frames)/time_run)) # Base: 30 fps; w/ alexnet: ~22 fps; # resnet50: ~6.5 fps ; # resnet152: ~3 fps

    #Commented capture feature
    #if key == 99: # on 'c' key press...
    #    saved_frame = frame
    #    mem_score = str(1.0) # Dummy value; replace with actual resnet output
    #    score_string = "Score: " + mem_score

    #    # cv2.line(frame,(0,0),(150,150),(255,255,255),15)
    #    cv2.putText(saved_frame, score_string, (10, 25), draw_font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    #    cv2.imshow("Memory Cam", saved_frame)

    #    while True: # Freeze frame until the user wants to continue
    #        break_key = cv2.waitKey(10)
    #        if break_key == 32: # Press space to continue
    #            break

    cv2.destroyWindow("Memory Cam")
    video_capture.release()


main()
