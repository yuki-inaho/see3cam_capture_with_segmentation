To run this software, it is necessary to ready 
camera setting file(cfg/camera_parameter.toml)

# Installation
1.  Downloading this repository.
```
git clone https://github.com/yuki-inaho/see3cam_capture_with_segmentation.git
```

2. Instaling dependent python libraries.
```
cd see3cam_capture_tool
pip install -r requirements.txt
```

# Usage
## Run capturing app
```
python capture.py
```

## Options
- "--toml-path", "-t": location of camera parameter file (default=`pwd`/cfg/camera_parameter.toml)
- "--directory-for-save", "-s": location to save images (default=`pwd`/data)
- "--config-name", "-c": location of inference configuration file (default=`pwd`/cfg/semantic_segmentation_multi_class.toml)
- "--rgb_rate", "-r": rgb:mask intensity rate of segmentation image (default=0.4)
- "--scale-for-visualization", "-sc": image resize rate for visualization (default=0.4)

