# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================
import sys
import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/aligned000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/aligned001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_colored = cv2.resize(cv2.imread("images/aligned001.jpg"), (0,0), fx=scale_factor,fy=scale_factor)
    print "Image shape" + str(ref_black.shape)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/aligned%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        scan_bits += bit_code * on_mask
        # TODO: populate scan_bits by putting the bit_code according to on_mask

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    correspondence_img = np.zeros((h,w,3), dtype=np.uint8)

    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code
            (x_p, y_p) = binary_codes_ids_codebook[scan_bits[y, x]]
            if x_p >= 1279 or y_p >= 799:
                continue
            projector_points.append((x_p, y_p))
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
            camera_points.append((x/2, y/2))
            y_p = 255 * y_p / (x_p + y_p)
            x_p = 255 * x_p / (x_p + y_p)

            correspondence_img[y,x,:] = np.array([0, y_p, x_p])


    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # converint into numpy array
    camera_points = np.reshape(np.array(camera_points, dtype=np.float32), [-1,1,2])
    projector_points = np.reshape(np.array(projector_points, dtype=np.float32), [-1,1,2])

    # Normalization (undistortion) of Points
    camera_points_normalized = cv2.undistortPoints(camera_points, camera_K, camera_d)
    projector_points_normalized = cv2.undistortPoints(projector_points, projector_K, projector_d)

    #camera_projection_mat = np.matmul(camera_K, np.eye(4)[:3])
    #projector_projection_mat = np.matmul(projector_K, np.concatenate((projector_R, projector_t), axis=1))

    camera_projection_mat = np.eye(4)[:3]
    projector_projection_mat = np.concatenate((projector_R, projector_t), axis=1)


    points_4d = cv2.triangulatePoints(projector_projection_mat, camera_projection_mat, projector_points_normalized, camera_points_normalized)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
    mask = np.stack((mask, mask, mask), axis=2)

    points_3d= np.reshape(np.extract(mask, points_3d), [-1, 1, 3] )
    points_3d_colored = []
    for i in range(points_3d.shape[0]):
        x = int(camera_points[i][0][0]*2)
        y = int(camera_points[i][0][1]*2)
        points_3d_colored.append([np.array(ref_colored[y][x][:])])


    points_3d_colored = np.array(points_3d_colored)
    points_3d_colored = np.concatenate((points_3d, points_3d_colored), axis = 2)

    cv2.imwrite('correspondence_img.jpg', correspondence_img)

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    # TODO: name the resulted 3D points as "points_3d"

    return points_3d, camera_points_normalized, projector_points_normalized, points_3d_colored

def write_3d_points(points_3d):

    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))

    return points_3d, camera_points, projector_points

def write_6d_points(points_6d):
    print("write output point cloud")
    print(points_6d.shape)
    output_name = sys.argv[1] + "output_color.xyzrgb"
    with open(output_name,"w") as f:
        for p in points_6d:
            f.write("%d %d %d %d %d %d\n"%(p[0,0],p[0,1],p[0,2],p[0,5],p[0,4],p[0,3]))

    return points_3d, camera_points, projector_points

if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d, camera_points, projector_points, points_3d_colored = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
    write_6d_points(points_3d_colored)

