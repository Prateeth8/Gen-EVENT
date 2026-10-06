# Gen-EVENT
A GAN based model to convert images to event frames for Visual Odometry in SLAM/ Robotics Applications.

The model details are uploaded to a google drive folder trained on DAVIS 240C dataset with GT being event frames and images being the input. There are 2 models: a) Pix2Pix and b) Cycle GAN.

The Pix2Pix architecture used a sobel edge based understanding for training the architecture whereas Cycle GAN used a direct image based understanding of event frames.

# Pipeline 

The Pix2Pix Training was performed as per the diagram:

![Pipeline](assets/pix2pix_gan_pipeline.png)

and the CYCLE GAN Training was performed as per the diagram:

![Pipeline](assets/cycle_gan_pipeline.png)

# Installation

Clone the Repo:

'''bash
git clone https://github.com/Prateeth8/Gen-EVENT.git

cd Gen-EVENT
'''

'''bash
python3 -m venv venv

source venv/bin/activate
'''

'''bash
pip install -r requirements.txt
'''

# Testing script

Model Paths access from Gdrive Link:

https://drive.google.com/drive/folders/1LKY3f-buFUC05VQ--OOdooPLGfLzgMcY?usp=sharing

Generating Event Frames from images

'''bash
python3 file_name.py --model_path model_path --data_path dataset_path --out_path output_dir_path
'''


# Results

Metric Results for both the models on boxes Rotation:

| Model | MAE ↓ | RMSE ↓ | SSIM ↑ |
|:------|------:|-------:|-------:|
| Pix2Pix | **0.16525** | 0.24280 | **0.20332** |
| CycleGAN | 0.17476 | **0.24233** | 0.17720 |

Model Output Plots:

![Results](assets/pix2pix_frame_000.png)
![Results](assets/cyclegan_frame_000.png)
