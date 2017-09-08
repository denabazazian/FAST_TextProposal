# FAST_TextProposal
- Combination of Text Proposals algorithm with Fully Convolutional Networks to efficiently reduce the number of proposals while maintaining the same recall level. <br />

## Installation

- This code requires:
   1. OpenCV 3.1 (tested with 02edfc8) 
   2. Caffe (tested with d21772c)
   3. tinyXML

- Installation:
  * cd EarlyPruning
  * cmake .
  * make

## Run
- Steps of generating proposals:
  1. Heatmaps: in order to generate the heatmaps, you can train your model and save your hatmaps according to this [link](https://github.com/denabazazian/pixelWise_textDetector).

  2. Early pruning: run the shell command for generating the proposals: <br />
 ```for i in {1..500}; do sh -c "echo 'Processing $i' && ./img2hierarchy /path/to/input/img_${i}.jpg /path/to/trained_boost_groups.xml /path/to/heatmap/img_${i}.png 0.14 > /path/to/proposals/img_$i.csv 2>/dev/null"; done```

## Evaluation
- Computing the confidences <br />
 ```confIoU.py prop2conf ./proposals/*.csv -threads=10```
- Computing the IoU <br />
  ```confIoU.py conf2IoU ./conf_proposals/*.csv -threads=10```
- Plot the detection rate <br />
   ```confIoU.py  '-extraPlotDirs={".":"proposals"}' getCumRecall ./conf_proposals/img_* '-IoUThresholds=[0.5]' -maxProposalsIoU=100000 -care=1```

## Citation
Please cite this work in your publications if it helps your research: <br />
@article{Bazazian17, <br />
author  = {Bazazian, Dena and Gomez, Raul and Nicolaou, Anguelos and Gomez, Lluis and Karatzas, Dimosthenis and Bagdanov, Andrew D.},<br />
title   = {FAST: Facilitated and Accurate Scene Text Proposals through FCN Guided Pruning},<br />
journal = {Pattern Recognition Letters(2017)},<br />
year    = {2017},<br />
ee      = {doi: 10.1016/j.patrec.2017.08.030 } <br />
}
