# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepStream-Yolo is an NVIDIA DeepStream SDK integration for YOLO object detection models. It provides custom C++/CUDA implementations for parsing and running various YOLO model variants (YOLOv5-v13, YOLO-NAS, RT-DETR, D-FINE, RF-DETR, and many others) on NVIDIA GPUs using TensorRT for optimized inference.

The project bridges YOLO models with DeepStream's GStreamer-based video analytics pipeline, enabling GPU-accelerated object detection on video streams.

## Build and Development Commands

### Compile Custom Library

The custom YOLO parsing library must be compiled before running inference:

```bash
# Set CUDA version according to DeepStream version
export CUDA_VER=12.8  # For DeepStream 8.0 on x86
export CUDA_VER=13.0  # For DeepStream 8.0 on Jetson

# Clean and build
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

**CUDA Version Reference:**
- x86: DS 8.0=12.8, DS 7.1=12.6, DS 7.0/6.4=12.2, DS 6.3=12.1, DS 6.2=11.8
- Jetson: DS 8.0=13.0, DS 7.1=12.6, DS 7.0/6.4=12.2, DS 6.3-6.1=11.4

### Optional Build Flags

```bash
# Build with OpenCV support (required for INT8 calibration)
OPENCV=1 make -C nvdsinfer_custom_impl_Yolo

# Build with performance graph support
GRAPH=1 make -C nvdsinfer_custom_impl_Yolo
```

### Run DeepStream Application

```bash
# Run with default config
deepstream-app -c deepstream_app_config.txt

# For YOLOv2/YOLOv2-Tiny (requires different config)
# Edit deepstream_app_config.txt to use config_infer_primary_yoloV2.txt
```

## Architecture Overview

### Core Components

1. **Custom TensorRT Engine Builder** (`nvdsinfer_custom_impl_Yolo/`)
   - `yolo.cpp/h`: Main YOLO model parser class that handles both Darknet (.cfg/.weights) and ONNX model formats
   - `nvdsinfer_yolo_engine.cpp`: Entry point for TensorRT engine creation via `NvDsInferYoloCudaEngineGet()`
   - `yoloPlugins.cpp/h`: Custom TensorRT plugins for YOLO-specific layers

2. **GPU-Accelerated Parsing** (`nvdsinfer_custom_impl_Yolo/`)
   - `nvdsparsebbox_Yolo_cuda.cu`: CUDA kernel implementation for GPU-based bounding box parsing
   - `nvdsparsebbox_Yolo.cpp`: CPU fallback bbox parser via `NvDsInferParseYolo()`
   - `yoloForward.cu` / `yoloForward_nc.cu` / `yoloForward_v2.cu`: CUDA kernels for different YOLO detection layer variants

3. **Layer Implementations** (`nvdsinfer_custom_impl_Yolo/layers/`)
   - Custom implementations for YOLO-specific layers: convolutional, deconvolutional, batchnorm, upsample, route, shortcut, pooling, reorg, SAM, implicit, channels, slice
   - Each layer implements TensorRT's IPluginV2 interface for integration into the inference engine

4. **Model Export Utilities** (`utils/`)
   - Python scripts for converting trained models to ONNX format compatible with DeepStream
   - One export script per model variant (e.g., `export_yoloV8.py`, `export_rtdetr_pytorch.py`)
   - Export scripts add DeepStream-specific output formatting layers

5. **INT8 Calibration** (`nvdsinfer_custom_impl_Yolo/`)
   - `calibrator.cpp/h`: TensorRT INT8 calibration implementation for Post-Training Quantization (PTQ)
   - Requires OpenCV support during build (`OPENCV=1`)

### Configuration Flow

The system uses a two-level configuration hierarchy:

1. **Application Config** (`deepstream_app_config.txt`)
   - Defines the DeepStream pipeline: sources, sinks, streammux, tiled-display, OSD
   - Points to inference config via `[primary-gie]` → `config-file` parameter

2. **Inference Config** (e.g., `config_infer_primary.txt`, `config_infer_primary_yoloV8.txt`)
   - Model-specific configuration: network type, input dimensions, batch size, precision (FP32/FP16/INT8)
   - For **Darknet models**: specifies `custom-network-config` (.cfg) and `model-file` (.weights)
   - For **ONNX models**: specifies `onnx-file` path
   - Always specifies: `custom-lib-path`, `parse-bbox-func-name`, `engine-create-func-name`
   - NMS parameters: `nms-iou-threshold`, `pre-cluster-threshold`, `topk` in `[class-attrs-all]`

### Model Processing Pipeline

```
YOLO Model (PyTorch/Darknet/Paddle)
    ↓
Export Script (utils/export_*.py) → adds DeepStream output layer
    ↓
ONNX Model / Darknet CFG+Weights
    ↓
Yolo Parser (yolo.cpp) → parses model architecture
    ↓
TensorRT Engine Builder → optimizes for target GPU
    ↓
.engine file (cached for reuse)
    ↓
DeepStream Inference (GPU-accelerated bbox parsing)
```

### Network Type Mapping

The `network-type` parameter in config files determines parsing behavior:

- Type 0: Standard YOLO (v2-v4, v7, v9, YOLOR)
- Type 1: YOLOX / PPYOLOE variants
- Type 2: RT-DETR / D-FINE / RF-DETR (transformer-based detectors)
- Type 3: YOLOv5 / v6 / v8 / v10 / v11 / v12 / v13 / YOLO-NAS

Each type uses different bbox decoding logic in the CUDA kernels.

## Model Integration Workflow

When adding a new YOLO model or debugging an existing one:

1. **Obtain the trained model** in its native framework format
2. **Export to ONNX** using the appropriate `utils/export_*.py` script (or create a new one for unsupported models)
3. **Copy ONNX and labels** to the DeepStream-Yolo root directory
4. **Select/modify config file**: use existing `config_infer_primary_*.txt` or create new one
   - Set correct `onnx-file`, `num-detected-classes`, `network-type`
   - Adjust `model-engine-file` naming pattern: `model_b{batch}_gpu{id}_{precision}.engine`
5. **Compile the library** (if modified): `make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo`
6. **Run inference**: `deepstream-app -c deepstream_app_config.txt`
   - First run generates TensorRT engine (can take 10+ minutes)
   - Subsequent runs load cached engine

## Important Implementation Details

### Dynamic vs Static Batching

- **Darknet models**: Dynamic batch-size enabled by default (set `force-implicit-batch-dim=0`)
- **ONNX models**: Typically use static batching unless explicitly exported with dynamic axes
- To force static batch for Darknet: set `force-implicit-batch-dim=1`

### GPU Post-Processing

This implementation uses GPU-accelerated bbox parsing (unlike basic DeepStream examples):

- Bounding box decoding, NMS, and filtering run on GPU via CUDA kernels
- Significantly faster than CPU post-processing for high-throughput scenarios
- Use `parse-bbox-func-name=NvDsInferParseYoloCuda` for GPU parsing (default in most configs)
- CPU fallback available via `parse-bbox-func-name=NvDsInferParseYolo`

### Cluster Mode

Always use `cluster-mode=2` for GPU-accelerated NMS. Other modes use CPU-based clustering.

### Engine File Caching

TensorRT engines are hardware-specific and cached as `.engine` files:

- Regenerate if changing: GPU model, TensorRT version, CUDA version, model file, precision mode, or batch size
- Engine generation can take 10+ minutes for large models
- DeepStream automatically generates engines on first run if not found

### Model-Specific Quirks

- **YOLOv2**: Requires separate config file (`config_infer_primary_yoloV2.txt`) due to different output format
- **RT-DETR variants**: Need specific export scripts per framework (PyTorch/Paddle/Ultralytics)
- **YOLO-NAS**: Has custom and standard variants with different output formats
- **YOLOX**: Legacy vs new versions require different export/config (see `config_infer_primary_yolox_legacy.txt`)

## Testing and Validation

The default `deepstream_app_config.txt` uses:
- **Source**: Sample video from DeepStream SDK (`/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4`)
- **Sink**: EGL display window (type=2)
- **Tiled display**: Single 1280x720 window

For testing custom models, modify `[source0]` to point to your video file/stream.

## Documentation Structure

Detailed model-specific instructions are in `docs/`:
- Individual model guides: `YOLOv5.md`, `YOLOv8.md`, `RTDETR_PyTorch.md`, etc.
- `customModels.md`: Generic instructions for integrating custom-trained models
- `INT8Calibration.md`: Post-training quantization workflow
- `dGPUInstalation.md`: Full dGPU platform setup instructions
- `multipleGIEs.md`: Running multiple YOLO models in parallel (secondary detectors)
- `benchmarks.md`: Performance comparisons across model variants

## Common Issues

1. **"CUDA_VER is not set"**: Must export CUDA_VER before compiling
2. **TensorRT engine generation hangs**: Normal for first run with large models, can take 10+ minutes
3. **"GLib pthread_setspecific error" on Ubuntu 22.04**: Known glib 2.0-2.72 bug, requires upgrading to glib 2.76 (see README.md Notes section)
4. **RTSP stream stuck at EOS**: Run `/opt/nvidia/deepstream/deepstream/update_rtpmanager.sh` to fix rtpjitterbuffer issue
5. **Build fails with missing headers**: Ensure DeepStream SDK is installed at `/opt/nvidia/deepstream/deepstream/`
6. **INT8 calibration fails**: Requires OpenCV support, rebuild with `OPENCV=1 make -C nvdsinfer_custom_impl_Yolo`
7. **Model not detected correctly**: Verify `network-type` parameter matches your model variant (see Network Type Mapping)
