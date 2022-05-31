import numpy as np
import cv2
import argparse
import sys
from calibration_store import load_stereo_coefficients
import matplotlib.pyplot as plt

import os

def Reprojection3D_multi(image, disparity1, disparity2, disparity3, Q1, Q2, Q3, output='stereo.ply'):
    
    # generate the 3D points for cam2 image from different pairs
    points_1 = cv2.reprojectImageTo3D(disparity1, Q1)
    mask_1 = disparity1 > disparity1.min()
    points_1[~mask_1] = 0

    points_2 = cv2.reprojectImageTo3D(disparity2, Q2)
    mask_2 = disparity2 > disparity2.min()
    points_2[~mask_2] = 0

    points_3 = cv2.reprojectImageTo3D(disparity3, Q3)
    mask_3 = disparity3 > disparity3.min()
    points_3[~mask_3] = 0

    # compine different pairs construction by smart avergining (by neglicting outlier points in each reconstruction) 
    mask_compined = np.array(mask_1, dtype=np.float16) + np.array(mask_2, dtype=np.float16) + np.array(mask_3, dtype=np.float16)
    print(np.unique(mask_compined))
    points_compine = (points_1+points_2+points_3) / np.expand_dims(mask_compined,axis=-1)

    desparity_compined = (disparity1+disparity2+disparity3) / mask_compined

    # get the final mask for the compination
    final_mask = np.logical_or(np.logical_or(mask_1, mask_2), mask_3)
    colors = image

    out_points = points_compine[final_mask]
    out_colors = image[final_mask]
    plt.imshow(points_compine[:,:,-1])
    plt.show()

    cv2.imwrite(f"{output.rsplit('/',1)[0]}/depth_compined.png", np.array(desparity_compined, dtype=np.uint8))

    # create the ply file
    verts = out_points.reshape(-1,3)
    colors = out_colors.reshape(-1,3)
    verts = np.hstack([verts, colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    with open(output, 'w') as f:
        f.write(ply_header %dict(vert_num = len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR) #.astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL) #.astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    _, filteredImg = cv2.threshold(filteredImg, 0, 5 * 16, cv2.THRESH_TOZERO)
    filteredImg = (filteredImg / 16).astype(np.uint8)

    return filteredImg


if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    # cam 1 caliberation with cam 2
    parser.add_argument('--calibration_file1', type=str, required=True, help='Path to the stereo calibration file')
    # cam 2 caliberation  with cam 3
    parser.add_argument('--calibration_file2', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--calibration_file3', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--cam1_source', type=str, required=True, help='Left video or v4l2 device name')
    parser.add_argument('--cam2_source', type=str, required=True, help='Right video or v4l2 device name')
    parser.add_argument('--cam3_source', type=str, required=True, help='Right video or v4l2 device name')
    parser.add_argument('--cam4_source', type=str, required=True, help='Right video or v4l2 device name')
    parser.add_argument('--output_folder', type=str, required=True, help='Right video or v4l2 device name')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # read the images for each camera
    cam1Frame = cv2.imread(args.cam1_source)
    cam1Frame = cv2.resize(cam1Frame, (int(cam1Frame.shape[1]/4), int(cam1Frame.shape[0]/4)))
    cam2Frame = cv2.imread(args.cam2_source)
    cam2Frame = cv2.resize(cam2Frame, (int(cam2Frame.shape[1]/4), int(cam2Frame.shape[0]/4)))
    cam3Frame = cv2.imread(args.cam3_source)
    cam3Frame = cv2.resize(cam3Frame, (int(cam3Frame.shape[1]/4), int(cam3Frame.shape[0]/4)))
    cam4Frame = cv2.imread(args.cam4_source)
    cam4Frame = cv2.resize(cam4Frame, (int(cam4Frame.shape[1]/4), int(cam4Frame.shape[0]/4)))


    # read the streo caliberation metrics between camera pairs (streo caliberation between (cam2, cam1), (cam2, cam3), (cam2, cam4) at our case)
    f1_K1, f1_D1, f1_K2, f1_D2, f1_R, f1_T, f1_E, f1_F, f1_R1, f1_R2, f1_P1, f1_P2, f1_Q = load_stereo_coefficients(args.calibration_file1)  # Get cams params
    f2_K1, f2_D1, f2_K2, f2_D2, f2_R, f2_T, f2_E, f2_F, f2_R1, f2_R2, f2_P1, f2_P2, f2_Q = load_stereo_coefficients(args.calibration_file2)  # Get cams params
    f3_K1, f3_D1, f3_K2, f3_D2, f3_R, f3_T, f3_E, f3_F, f3_R1, f3_R2, f3_P1, f3_P2, f3_Q = load_stereo_coefficients(args.calibration_file3)  # Get cams params



    height, width, channel = cam1Frame.shape  # We will use the shape for remap

    # Undistortion and Rectification part!
    # for first streo pair
    cam1MapX, cam1MapY = cv2.initUndistortRectifyMap(f1_K2, f1_D2, f1_R2, f1_P2, (width, height), cv2.CV_32FC1)
    cam1_rectified = cv2.remap(cam1Frame, cam1MapX, cam1MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    
    cam2MapX_1, cam2MapY_1 = cv2.initUndistortRectifyMap(f1_K1, f1_D1, f1_R1, f1_P1, (width, height), cv2.CV_32FC1)
    cam2_rectified_1 = cv2.remap(cam2Frame, cam2MapX_1, cam2MapY_1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # for second streo pair
    cam2MapX_2, cam2MapY_2 = cv2.initUndistortRectifyMap(f2_K1, f2_D1, f2_R1, f2_P1, (width, height), cv2.CV_32FC1)
    cam2_rectified_2 = cv2.remap(cam2Frame, cam2MapX_2, cam2MapY_2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    cam3MapX, cam3MapY = cv2.initUndistortRectifyMap(f2_K2, f2_D2, f2_R2, f2_P2, (width, height), cv2.CV_32FC1)
    cam3_rectified = cv2.remap(cam3Frame, cam3MapX, cam3MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # for third stereo pair
    cam2MapX_3, cam2MapY_3 = cv2.initUndistortRectifyMap(f3_K1, f3_D1, f3_R1, f3_P1, (width, height), cv2.CV_32FC1)
    cam2_rectified_3 = cv2.remap(cam2Frame, cam2MapX_3, cam2MapY_3, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    cam4MapX, cam4MapY = cv2.initUndistortRectifyMap(f3_K2, f3_D2, f3_R2, f3_P2, (width, height), cv2.CV_32FC1)
    cam4_rectified = cv2.remap(cam4Frame, cam4MapX, cam4MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # We need grayscale for disparity map.
    gray_cam1 = cv2.cvtColor(cam1_rectified, cv2.COLOR_BGR2GRAY)
    gray_cam2_1 = cv2.cvtColor(cam2_rectified_1, cv2.COLOR_BGR2GRAY)
    gray_cam2_2 = cv2.cvtColor(cam2_rectified_2, cv2.COLOR_BGR2GRAY)
    gray_cam2_3 = cv2.cvtColor(cam2_rectified_3, cv2.COLOR_BGR2GRAY)
    gray_cam3 = cv2.cvtColor(cam3_rectified, cv2.COLOR_BGR2GRAY)
    gray_cam4 = cv2.cvtColor(cam4_rectified, cv2.COLOR_BGR2GRAY)

    # calcaulate disparity map between ecah image pair (consdring Cam 2 image as the main image for all pairs)
    disparity_image_1 = depth_map(gray_cam2_1, gray_cam1)  # Get the disparity map for the second camera
    disparity_image_2 = depth_map(gray_cam2_2, gray_cam3)  # Get the disparity map for the second camera
    disparity_image_3 = depth_map(gray_cam2_3, gray_cam4)  # Get the disparity map for the second camera

    # inverse rectify to get the disparity map for the orginal image not the rectified one
    inv_cam2MapX_1, inv_cam2MapY_1 = cv2.initInverseRectificationMap(f1_K1, f1_D1, f1_R1, f1_P1, (width, height), cv2.CV_32FC1)
    disparity_1 = cv2.remap(disparity_image_1, inv_cam2MapX_1, inv_cam2MapY_1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    inv_cam2MapX_2, inv_cam2MapY_2 = cv2.initInverseRectificationMap(f2_K1, f2_D1, f2_R1, f2_P1, (width, height), cv2.CV_32FC1)
    disparity_2 = cv2.remap(disparity_image_2, inv_cam2MapX_2, inv_cam2MapY_2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    inv_cam2MapX_3, inv_cam2MapY_3 = cv2.initInverseRectificationMap(f3_K1, f3_D1, f3_R1, f3_P1, (width, height), cv2.CV_32FC1)
    disparity_3 = cv2.remap(disparity_image_3, inv_cam2MapX_3, inv_cam2MapY_3, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # plt.imshow(gray_cam1, "gray")
    # plt.show() 
    # plt.imshow(gray_cam2_1, "gray")
    # plt.show()
    # plt.imshow(gray_cam2_2, "gray")
    # plt.show()
    # plt.imshow(gray_cam3, "gray")
    # plt.show()
    # plt.imshow(gray_cam2_3, "gray")
    # plt.show()
    # plt.imshow(gray_cam4, "gray")
    # plt.show()
    # plt.imshow(disparity_image_1, 'gray')
    # plt.show()
    # plt.imshow(disparity_1, 'gray')
    # plt.show()
    # plt.imshow(disparity_image_2, 'gray')
    # plt.show()
    # plt.imshow(disparity_2, 'gray')
    # plt.show()
    # plt.imshow(disparity_image_3, 'gray')
    # plt.show()
    # plt.imshow(disparity_3, 'gray')
    # plt.show()
    cv2.imwrite(f"{args.output_folder}/cam1_rectified.png", cam1_rectified)
    cv2.imwrite(f"{args.output_folder}/cam2_rectified_1.png", cam2_rectified_1)
    cv2.imwrite(f"{args.output_folder}/cam2_rectified_2.png", cam2_rectified_2)
    cv2.imwrite(f"{args.output_folder}/cam2_rectified_3.png", cam2_rectified_3)
    cv2.imwrite(f"{args.output_folder}/cam3_rectified.png", cam3_rectified)
    cv2.imwrite(f"{args.output_folder}/cam4_rectified.png", cam4_rectified)
    cv2.imwrite(f"{args.output_folder}/disparity_1.png", disparity_1)
    cv2.imwrite(f"{args.output_folder}/disparity_2.png", disparity_2)
    cv2.imwrite(f"{args.output_folder}/disparity_3.png", disparity_3)

    # use general Q matrix to combine between all the camera to have the same depth range

    # Construct the Q matric for the 3D construction
    # get the focalLength and the principal Points for the camera

    baseline = 193.001/2
    f = 3979.911/2
    Q = np.array([[1, 0, 0, -2964/2], [0, 1, 0, -2000/2],[0, 0, 0, f],[0, 0, -1/baseline, -124.343/baseline]])
    

    Reprojection3D_multi(cam2Frame, disparity_1, disparity_2, disparity_3, Q, Q, Q, f"{args.output_folder}/3DObj.ply")



    

    