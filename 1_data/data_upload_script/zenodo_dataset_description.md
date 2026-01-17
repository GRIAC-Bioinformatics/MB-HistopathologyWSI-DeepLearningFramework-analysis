# Histopathology Whole Slide Image Dataset: Lung Tissue Analysis

## Dataset Overview

The study used 162 whole-slide images (WSIs) of H&E-stained cross-sectional airway sections from 75 patients obtained from the archive of the Department of Pathology and Medical Biology at the University Medical Center Groningen.

## Patient Population

Patients who underwent surgical lung resection for COPD (lung transplants)
Patients who had lung resection for lung cancer, with either COPD GOLD stage 1 or 2 or normal lung function
Patient characteristics including smoking history and lung function are documented in Supplementary Table 1

## Image Acquisition

Glass slides were scanned using a Philips Ultra Fast Scanner 1.6 at 40x magnification
Single-focus layer without Z-stacking
Automatic tissue detection with focus points for optimal imaging
WSIs converted to BigTIFFs with resolution of 0.25 micrometres per pixel (mpp)
All WSIs contained noncartilaginous airways (bronchioles) far distant from cancer involvement

## Annotations

Manual annotations of airway walls performed in QuPath (version 0.2.0)
Annotations included respiratory epithelium and airway smooth muscle marking boundaries of two key lung compartments: submucosa and adventitia
Annotation performed primarily by a digital pathology technician and validated by a pathologist

## Data Processing

Images split into 120 × 120 pixel patches using sliding window method
Data cleaning included masking of cellular structures (RBCs, lymphocytes, epithelium), artefacts (white backgrounds, carbon particles)

## Contents

This dataset contains the following components:

### wsi.zip
- Complete set of 162 whole slide images in BigTIFF format
- Resolution: 0.25 μm/pixel
- Format: .tiff files

### patches_120x120.zip
- Image patches extracted from WSIs
- Size: 120 × 120 pixels
- Tissue compartments: Inside (submucosa) and Outside (adventitia) airway walls

### annotation_data_qupath.zip
- Manual annotations created in QuPath
- Includes airway wall boundaries and tissue compartment definitions
- Project files and annotation data for all 162 WSIs

### classifiers_qupath.zip
- Trained classifiers for automated tissue detection
- Includes epithelium and smooth muscle classifiers

## Ethical Considerations

Study used archived lung tissue from leftover surgical material
Approved by UMCG Central Ethics Review Board (Research Register number: 202200107)
All patient data were fully de-identified prior to analysis

## Citation

If you use this dataset, please cite:

```bibtex
@article{vanbreugel2024,
  title={A deep learning framework for histopathological analysis of pixel-level extracellular matrix variation in standard H&E-stained images},
  author={van Breugel, Merlijn and de Jong, Esm{\\'e}e and Buikema, Henk J. and Petoukhov, Ilya and Nawijn, Martijn C. and Burgess, Janette K. and Timens, Wim},
  journal={[Journal Name]},
  year={2026},
  note={Code available at: [repository URL]}
}
```

## Contact

For questions about this dataset:

**Merlijn van Breugel**
- Email: merlijnvanbreugel@gmail.com
- UMCG: m.van.breugel@umcg.nl

## License

This dataset is available under Creative Commons Attribution 4.0 International (CC BY 4.0)