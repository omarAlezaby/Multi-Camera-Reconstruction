# Multi-camera Reconstuction using 4 cameras

This project is For Multi-camera Reconstuction using 4 Go-pro Cameras

## Usage

### Single camera calibration
First step is to do camera caliberation for each camera in the setup sepritly.

Example usage:

```bash
python single_camera_calibration.py data\Cam1_Caliberation --image_format JPG --prefix l --square_size 0.025 --width 9 --height 6 --save_file Cam1_calib.yml
```

### Stereo camera calibration
Second step is stereo caliberation by specifing a main camera and do a stereo calibeation between each camera and the other cameras.
For our case Cam 2 in the main camera and we made stereo caliberation for (cam2, cam1), (cam2, cam3), and (cam2, cam3) pairs

Example usage:

```bash
python stereo_camera_calibration.py --left_file Cam2_calib.yml --right_file Cam1_calib.yml --left_prefix C2_ --right_prefix C1_ --left_dir data\Cam2_Caliberation --right_dir data\Cam1_Caliberation --image_format JPG --square_size 0.025 --save_file C1_C1_stereo_calib.yml
```

### Disparity map

Example usage:
For reconstuction from stereo setup
```bash
python Stereo_reconstruction.py --calibration_file  stereo_cam_test.yml --left_source data\Capture_4\object_left\l24.JPG --right_source data\Capture_4\object_right\r24.JPG --output_folder .\caliberation_last_streo_l1\face_L3\       

```

For multi-camera Reconstuction

```bash
python Mulit_construction.py --calibration_file1 C2_C1_cam_test.yml --calibration_file2 C2_C3_cam_test.yml --calibration_file3 C2_C4_cam_test.yml --cam1_source data\Multi_Camera\Obj\Cam1_obj\C1_6.JPG --cam2_source data\Multi_Camera\Obj\Cam2_obj\C2_6.JPG --cam3_source data\Multi_Camera\Obj\Cam3_obj\C3_6.JPG --cam4_source data\Multi_Camera\Obj\Cam4_obj\C4_6.JPG --output_folder caliberation_multi_l1\bottle_L1    
```

# Refrences
- https://github.com/aliyasineser/stereoDepth
- https://github.com/FlagArihant2000/stereo
- [Metric 3D Surface Reconstruction from
Uncalibrated Image Sequences](https://www.researchgate.net/profile/Marc-Pollefeys/publication/2428849_Metric_3D_Surface_Reconstruction_from_Uncalibrated_Image_Sequences/links/53f5d0b70cf22be01c3fb159/Metric-3D-Surface-Reconstruction-from-Uncalibrated-Image-Sequences.pdf)