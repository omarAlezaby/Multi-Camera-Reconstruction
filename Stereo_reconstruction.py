import numpy as np
import cv2
import argparse
import sys
from calibration_store import load_stereo_coefficients
import matplotlib.pyplot as plt
import os

def Reprojection3D_specific(image, disparity, Q, output):
    
    points = cv2.reprojectImageTo3D(disparity, Q)
    plt.imshow(points[:,:,-1])
    plt.show()
    
    mask = disparity > disparity.min()
    colors = image

    out_points = points[mask]
    out_colors = image[mask]

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
    # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    # filteredImg = np.uint8(filteredImg)
    _, filteredImg = cv2.threshold(filteredImg, 0, 5 * 16, cv2.THRESH_TOZERO)
    filteredImg = (filteredImg / 16).astype(np.uint8)

    return filteredImg


if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--left_source', type=str, required=True, help='Left video or v4l2 device name')
    parser.add_argument('--right_source', type=str, required=True, help='Right video or v4l2 device name')
    parser.add_argument('--output_folder', type=str, required=True, help='output folder')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    leftFrame = cv2.imread(args.left_source)
    leftFrame = cv2.resize(leftFrame, (int(leftFrame.shape[1]/4), int(leftFrame.shape[0]/4)))
    rightFrame = cv2.imread(args.right_source)
    rightFrame = cv2.resize(rightFrame, (int(rightFrame.shape[1]/4), int(rightFrame.shape[0]/4)))



    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params

    

    height, width, channel = leftFrame.shape  # We will use the shape for remap

    # get the focalLength and the principal Points for the camera
    cam1_fovx, cam1_fovy, cam1_focalLength, cam1_principalPoint, cam1_aspectRatio = cv2.calibrationMatrixValues(K1, (width, height), 6.17, 4.55)

    cam2_fovx, cam2_fovy, cam2_focalLength, cam2_principalPoint, cam2_aspectRatio = cv2.calibrationMatrixValues(K2, (width, height), 6.17, 4.55)


    # # Undistortion and Rectification part!
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # We need grayscale for disparity map.
    gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    disparity_image = depth_map(gray_left, gray_right)  # Get the disparity map

    # inverse the rectification to get the dsiparity map to the original image not the rectified one
    inv_leftMapX, inv_leftMapY = cv2.initInverseRectificationMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    disparity_left = cv2.remap(disparity_image, inv_leftMapX, inv_leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    plt.imshow(left_rectified)
    plt.show() 
    plt.imshow(right_rectified)
    plt.show()
    plt.imshow(disparity_image, 'gray')
    plt.show()
    plt.imshow(disparity_left, 'gray')
    plt.show()

    cv2.imwrite(f"{args.output_folder}/left_rectified.png", left_rectified)
    cv2.imwrite(f"{args.output_folder}/right_rectified.png", right_rectified)
    cv2.imwrite(f"{args.output_folder}/disparity_image.png", disparity_image)
    cv2.imwrite(f"{args.output_folder}/disparity_left.png", disparity_left)


    # Construct the Q matric for the 3D construction
    b = T[0][0]

    Q = np.array([[1, 0, 0, -cam1_principalPoint[0]*1000], 
    [0, 1, 0, -cam1_principalPoint[1]*1000], 
    [0, 0, 0, cam1_focalLength*1000], 
    [0, 0, -1/(b*1000), (cam1_principalPoint[0]-cam2_principalPoint[0])/b]])

    # 3D construction
    Reprojection3D_specific(leftFrame, disparity_left, Q, f"{args.output_folder}/3Dobj.ply")
    

    