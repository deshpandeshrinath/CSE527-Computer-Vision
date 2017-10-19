import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier(
    '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')

def help_message():
    print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
    print("[Question Number]")
    print("1 Camshift")
    print("2 Particle Filter")
    print("3 Kalman Filter")
    print("4 Optical Flow")
    print("[Input_Video]")
    print("Path to the input video")
    print("[Output_Directory]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]


def particle_evaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]


def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c, r, w, h = window
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi,
                       np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


def camShift_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        print "No Video Found!"
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)
    track_window = (c,r,w,h)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (0,c+w/2.0,r+h/2.0))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # calculate the HSV histogram in the window
    roi_hist = hsv_histogram_for_window(frame, (c, r, w,
                                                h))  # this is provided for you
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while (1):
        ret, frame = v.read()  # read another frame
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # Perform tracking using CamShift
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # center of rectangel
            pt = (frameCounter, ret[0][0], ret[0][1])
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,2)
            cv2.imshow('img2',img2)

            # write the result to the output file
            output.write("%d,%d,%d\n" % pt)  # Write as frame_index,pt_x,pt_y
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
        else:
            break
        frameCounter = frameCounter + 1

    output.close()


def particleFilter_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (0, c+w/2.0, r+h/2.0)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    # initialize the tracker
    n_particles = 500

    # weights are uniform (at first)
    weights = np.ones(n_particles) / n_particles
    # Initial position
    init_pos = np.array([c + w/2.0,r + h/2.0], int)
    # Init particles to init position
    particles = np.ones((n_particles, 2), int) * init_pos
    #np.add(particles, np.random.randn(n_particles,2),
    #       out=particles, casting="unsafe")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    # Evaluate appearance model
    f0 = particle_evaluator(hist_bp, particles.T) * np.ones(n_particles)

    pos = np.sum(particles.T * weights, axis=1).astype(int)

    stepsize = 8
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        # # Particle motion model: uniform step (TODO: find a better motion model)
        np.add(particles, np.random.randn(n_particles,2)*stepsize,
               out=particles, casting="unsafe")
        # np.add(particles, np.random.uniform(-stepsize, stepsize,
        #    particles.shape), out=particles, casting="unsafe")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        im_h, im_w, layers = frame.shape
        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

        f = particle_evaluator(dst, particles.T) # Evaluate particles
        # print("f is :\n" + str(f))
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        # print weights
        weights /= np.sum(weights)                  # Normalize w
        # expected position: weighted average
        # If particle cloud degenerate:

        if 1. / np.sum(weights**2) < n_particles / 2.:
            #Resample particles according to weights
            particles = particles[resample(weights),:]

        pos = np.sum(particles.T * weights, axis=1).astype(int)

        img2 = cv2.circle(frame,(pos[0], pos[1]), 5, (0,0,255), -1)
        cv2.imshow("img2",img2)
        cv2.waitKey(27)
        pt = (frameCounter, pos[0], pos[1])
        # use the tracking result to get the tracking point (pt):
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


def kalmanFilter_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (0, c+w/2.0, r+h/2.0)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    # initialize the tracker
    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman = cv2.KalmanFilter(4,2,0) # 4 state/hidden, 2 measurement, 0 control
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                    [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)      # you can tweak these to make the tracker
    kalman.processNoiseCov = 1e-3 * np.eye(4, 4)      # respond faster to change and be less smooth
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        # perform the tracking
        prediction = kalman.predict()
        pt = prediction.reshape(4,)

        # State Measurement
        c, r, w, h = detect_one_face(frame)
        measurement = np.array([c+w/2.0, r+h/2.0], dtype='float64')
        if (w != 0 and h!= 0): # e.g. face found
            pt = kalman.correct(measurement)
        op = (frameCounter ,pt[0], pt[1])
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
        output.write("%d,%d,%d\n" % op) # Write as frame_index,pt_x,pt_y
        img_int = cv2.rectangle(frame,(c,r),(c+w,r+h),(0,255,0),1)
        img2 = cv2.circle(img_int,(int(pt[0]), int(pt[1])), 5, (0,0,255), -1)
        cv2.imshow('img_kalman_filter',img2)
        #fourcc = cv2.VideoWriter_fourcc(*'H264')
        #print img2.shape
        #out_video = cv2.VideoWriter('kalman_filter_1.mp4', -1, 20.0, (120, 160))
        # img2 = cv2.flip(img2,0)
        #out_video.write(img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        frameCounter = frameCounter + 1

    output.close()


def opticalFlow_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (0, c+w/2.0, r+h/2.0)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.05,
                       minDistance = 1,
                       blockSize = 7 )
    init_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_mask = np.zeros(init_gray.shape, dtype=np.uint8)
    face_mask[r:r+h, c:c+w] = 1
    p0 = cv2.goodFeaturesToTrack(init_gray, mask=face_mask, **feature_params)
    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    color = np.random.randint(0,255,(100,3))
    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(init_gray,
                    frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        pos = np.mean(good_new, axis=0)
        pt = (frameCounter, pos[0], pos[1])
        # draw the tracks
        # for i,(new,old) in enumerate(zip(good_new,good_old)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 1)
        #     frame = cv2.circle(frame,(a,b),1,color[i].tolist(),-1)
        frame = cv2.circle(frame, (pos[0], pos[1]), 5, [0, 0, 255], -1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        init_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        # write the result to the output file
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2])
    if (video.isOpened() == False):
        raise ("Error opening video stream or file")

    if (question_number == 1):
        camShift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        particleFilter_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kalmanFilter_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        opticalFlow_tracker(video, "output_of.txt")
'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

# --- tracking
'''