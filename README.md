# 🌙 LUNAR IMAGE REGISTRATION

## 🚀 AI-Powered Cross-Modal Lunar Image Registration System

A smart lunar image registration system designed to identify accurate corresponding points between source and reference lunar images and align them into a common coordinate system using AI-based feature matching, geometric verification, and image registration.

The system is designed to handle challenging lunar imagery with differences in illumination, resolution, viewing angle, scale, and sensor characteristics using LoFTR, illumination-robust feature processing, RANSAC filtering, and adaptive tie-point refinement. 🌙🛰️

The system supports lunar imagery from multiple sensors and incorporates DEM and physical-geometry information when the required geometric information is available. 🔭📐

---

# ✨ Key Features

✅ Cross-Modal Lunar Image Matching  
✅ AI-Based Tie-Point Detection  
✅ LoFTR Feature Matching  
✅ Illumination-Robust Feature Processing  
✅ Multi-Scale Image Representation  
✅ Candidate Correspondence Filtering  
✅ Uniform Grid-Based Tie-Point Selection  
✅ RANSAC Outlier Filtering  
✅ Confidence-Based Match Evaluation  
✅ Adaptive Sub-Pixel Refinement  
✅ DEM-Assisted Geometric Validation  
✅ Image Registration and Alignment  
✅ Quantitative Registration Evaluation  
✅ Tie-Point Visualization  
✅ Registered Image Generation  
✅ Overlay and Difference Visualization  
✅ MongoDB-Based Result Storage  

---

# 🛠️ Technologies Used

## 🤖 Artificial Intelligence & Computer Vision

- PyTorch
- LoFTR
- OpenCV
- NumPy
- SIFT
- RANSAC

## 🌍 Geospatial & Lunar Data Processing

- GDAL
- DEM / Elevation Data
- Lunar Image Geometry
- Coordinate Transformation

## ⚙️ Backend

- Python
- FastAPI

## 🎨 Frontend

- React
- JavaScript
- HTML
- CSS

## 🗄️ Database

- MongoDB

## 🔧 Development Tools

- Git
- GitHub
- VS Code

---

# ⚙️ System Workflow

1️⃣ User provides a source lunar image and a reference lunar image

2️⃣ System performs input and metadata/context checks

3️⃣ Relevant DEM and geographic information are retrieved when available

4️⃣ Images are preprocessed and represented at multiple scales

5️⃣ Illumination-robust features are extracted

6️⃣ LoFTR identifies candidate correspondences between the images

7️⃣ Candidate matches are filtered and distributed using uniform grid selection

8️⃣ RANSAC removes geometrically inconsistent matches

9️⃣ Match confidence is evaluated

🔟 Tie points are refined toward sub-pixel accuracy

1️⃣1️⃣ DEM/physical geometry is used for validation when the required information is available

1️⃣2️⃣ Final reliable tie points are generated

1️⃣3️⃣ The source image is registered to the reference image

1️⃣4️⃣ Registration quality is evaluated using quantitative metrics

1️⃣5️⃣ Final tie points, registered image, overlay, and evaluation results are generated

---

# 🔬 Core Modules

🔹 Image Input & Validation  
🔹 Metadata & Context Analysis  
🔹 DEM Retrieval  
🔹 Multi-Scale Representation  
🔹 Illumination-Robust Feature Extraction  
🔹 LoFTR Matching  
🔹 Candidate Correspondence Filtering  
🔹 Uniform Grid Tie-Point Selection  
🔹 RANSAC Outlier Filtering  
🔹 Confidence Evaluation  
🔹 Sub-Pixel Refinement  
🔹 Geometric / DEM Validation  
🔹 Image Registration  
🔹 Quantitative Validation  
🔹 Result Storage  

---

# 📊 Evaluation Metrics

The system evaluates registration quality using:

- Number of source keypoints
- Number of reference keypoints
- Candidate matches
- RANSAC inliers
- RANSAC outliers
- Inlier ratio
- Mean reprojection error
- Median reprojection error
- Reprojection RMSE
- Processing time

These metrics help determine the reliability and accuracy of the final registration result. 📈

---

# 📦 Final Outputs

The system generates:

🔗 Final Tie Points  
🖼️ Registered Source Image  
🔍 Tie-Point Visualization  
🌗 Overlay / Difference Image  
📊 Registration Metrics  
📄 Tie-Point Data  
📋 Processing Report  

---

# 🎯 Project Objectives

- Improve correspondence between multi-sensor lunar images 🌙
- Handle illumination and resolution differences
- Generate spatially distributed tie points
- Remove unreliable feature correspondences
- Improve tie-point localization through refinement
- Use terrain and physical geometry when available
- Produce accurately registered lunar imagery
- Provide measurable registration quality through quantitative evaluation

---

# 💡 Why This System?

Traditional feature matching can struggle when two lunar images have significant differences in illumination, resolution, viewpoint, or sensor characteristics.

This system combines **AI-based matching, geometric consistency, spatially distributed tie points, refinement, and physical/terrain information** to provide a more reliable lunar image registration workflow. 🛰️🌙

---

# 📌 Conclusion

This project demonstrates an AI-driven lunar image registration pipeline for establishing reliable correspondences between multi-sensor lunar imagery. By combining LoFTR-based matching, illumination-robust processing, uniform tie-point selection, RANSAC filtering, sub-pixel refinement, geometric validation, and image registration, the system provides a complete workflow for accurate lunar image alignment.

The architecture is designed to support challenging cross-modal lunar imagery while providing quantitative metrics and visual outputs for evaluating registration quality. 🚀🔭

---

## ⭐ Lunar Imaging • AI Matching • Tie Points • Image Registration
