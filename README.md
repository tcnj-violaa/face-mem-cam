##face_mem_cam

NOTE: This is archived code from 2019, cleaned up slightly for readability. The actual functionality is not guaranteed to be functioning in its current state, especially in 2022.  

face_mem_cam is an application that predicts the memorability of video frames from a webcam, in real-time,
as well as performing facial detection and facial landmarking. These functions are displayed graphically as 
the app is running -- ie., the predicted memorability score is displayed both numerically and graphically via a meter,
boundary boxes are drawn around detected faces, and feature-based landmark points are drawn on those faces.

-- Memorability Prediction Pipeline --

Our memorability prediction model contains two discrete parts: a pretrained resnet model, and a custom pretrained rnn model. Typically, resnet is
an image classifying model: given a picture, it tries to determine the class that the image belongs to. Our output is a set of probabilities
that our input 'is' any one of those classes. For example, if we pass an image of an iguana to the model, it is likely to return a high
probability of it being of the 'common_iguana' class, and a very low probablity of it being of the 'cell_phone' class.

We use the resnet model in a slightly modified way, however. For this task, the classification itself is not as useful as what the model ultimately
has learned and uses to calculate these classification probabilities. The layer just prior to the final classification layer contains rich, high-level
image content information that we want to use -- to do so, we simply remove that final classification layer from the model. Now, when we pass an image through the
model, our output is -- instead of a classification -- a 2048-dimensional vector representing all of this valuable information.

To use this information to predict a memorability score, then, we pass it as input into our rnn-based regression model. An rnn's output captures a representation of 
temporal information -- the given features and how they change across time -- so we must give it input that accounts for this added time dimension. Video naturally
lends itself to this: a video frame is a natural 'time step.' In our case, we can simply capture three frames and pass each through the resnet model, so we ultimately have a three-set
of per-frame 2048-dimensional outputs. We pass this set into the rnn, it looks over each, and returns a linear representation of the information over this short time period. That is then
used for simple regression to the memorability score. This process is repeated every time three frames are captured from the webcam, that is to say, very often. For a 30 fps webcam, that's
every tenth of a second.

-- Face Detection and Landmarking --

To accomplish both face detection and landmarking, we use out-of-the-box algorithms from various libraries. OpenCV2 (imported as cv2) offers a Haar cascade
classifier, pretrained to quickly detect frontal faces contained in a larger image: in our case, each video frame. For each detected face, the cascade
returns the rectangular bounds that contain them. These bounds are important for our face landmark detector, offered by dlib -- it looks at the
image inside the spaces defined by the bounds (faces, hopefully) and runs a human face shape predictor over it. From the detector we get a neat set of
points defining various facial features: eyebrows, eyes, nose, lips, and jawline.

In short summary, a Haar cascade finds the faces in the frame, and our facial landmark detector 'parses' each face.

-- Planned Features --

Ultimately, our goal is to use the facial landmarks we obtain as additional features in our memorability score prediction model. This will require dataset
design, feature engineering, and the modification/retraining of our prediction model.
