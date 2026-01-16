setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.58389 0.72169 0.37179 ", "Stain 2" : "Eosin", "Values 2" : "0.20589 0.91946 0.33496 ", "Background" : " 225 223 229 "}');
clearDetections();
selectObjectsByClassification("Focus");
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield": "Optical density sum",  "requestedPixelSizeMicrons": 0.5,  "backgroundRadiusMicrons": 8.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.1,  "maxBackground": 2.0,  "watershedPostProcess": true,  "cellExpansionMicrons": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');
runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin', '{"fwhmMicrons": 25.0,  "smoothWithinClasses": false}');
runClassifier('N:\\Werkstudenten\\1. Eigen mappen\\Esmee\\UMCG\\Qupath_annotation_def\\classifiers\\Classifier 3.qpclassifier');