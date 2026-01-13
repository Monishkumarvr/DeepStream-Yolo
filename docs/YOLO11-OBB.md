# YOLO11-OBB usage

**NOTE**: YOLO11-OBB (Oriented Bounding Box) models are used for detecting rotated objects. The OBB parser converts oriented boxes to axis-aligned bounding boxes (AABB) for DeepStream visualization.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolo11_obb file](#edit-the-config_infer_primary_yolo11_obb-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLO11 repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -e .
pip3 install onnx onnxslim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolo11_obb.py` file from `DeepStream-Yolo/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [YOLO11-OBB](https://github.com/ultralytics/assets/releases/) releases (example for YOLO11n-OBB)

```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt
```

**NOTE**: You can use your custom OBB model trained on datasets like DOTAv1, DOTAv1.5, or DOTAv2.

#### 4. Convert model

Generate the ONNX model file (example for YOLO11n-OBB)

```
python3 export_yolo11_obb.py -w yolo11n-obb.pt --dynamic
```

**NOTE**: To change the inference size (default: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1024

```
-s 1024
```

or

```
-s 1024 1024
```

**NOTE**: To simplify the ONNX model (DeepStream >= 6.0)

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```

**NOTE**: If you are using the DeepStream 5.1, remove the `--dynamic` arg and use opset 12 or lower. The default opset is 17.

```
--opset 12
```

#### 5. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo` folder.

##

### Compile the lib

1. Open the `DeepStream-Yolo` folder and compile the lib

2. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  DeepStream 5.1 = 11.1
  ```

* Jetson platform

  ```
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 / 5.1 = 10.2
  ```

3. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

##

### Edit the config_infer_primary_yolo11_obb file

Edit the `config_infer_primary_yolo11_obb.txt` file according to your model (example for YOLO11n-OBB with 15 classes)

```
[property]
...
onnx-file=yolo11n-obb.onnx
...
num-detected-classes=15
...
parse-bbox-func-name=NvDsInferParseYoloOBB
...
```

**NOTE**: For GPU-accelerated parsing (recommended for better performance), use:

```
[property]
...
parse-bbox-func-name=NvDsInferParseYoloOBBCuda
...
```

**NOTE**: The **YOLO11-OBB** resizes the input with center padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```

**NOTE**: OBB models output oriented bounding boxes with rotation angles. The parser converts these to axis-aligned bounding boxes (AABB) that fully enclose the rotated objects for visualization in DeepStream. The original angle information is lost in this conversion.

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yolo11_obb.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.

##

### Understanding OBB Output Format

YOLO11-OBB models output the following format per detection:

- **x_center, y_center**: Center coordinates of the oriented box
- **width, height**: Dimensions of the oriented box
- **class_probabilities**: Probability for each class (DOTAv1 has 15 classes)
- **angle**: Rotation angle in radians (range: 0 to π/2)

The DeepStream parser (`NvDsInferParseYoloOBB` or `NvDsInferParseYoloOBBCuda`) converts each oriented box to an axis-aligned bounding box using the formula:

```
half_aabb_w = (width * |cos(angle)| + height * |sin(angle)|) / 2
half_aabb_h = (width * |sin(angle)| + height * |cos(angle)|) / 2
```

This ensures the axis-aligned box fully encloses the rotated object.

##

### Common OBB Datasets

YOLO11-OBB models are typically trained on:

- **DOTAv1**: 15 classes (plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool)
- **DOTAv1.5**: 16 classes (adds container-crane)
- **DOTAv2**: 18 classes (adds airport and helipad)

Make sure `num-detected-classes` matches your model's training dataset.
