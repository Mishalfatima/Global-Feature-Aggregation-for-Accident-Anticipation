# Global-Feature-Aggregation-for-Accident-Anticipation
Anticipating accident in Dashcam videos using Global Feature Aggregation

The code aims at anticipating accidents few seconds before they actually occur by aggregating features globally in a frame. 
The extract_features.py file extracts features using VGG-16 from the objects and frames in a video sequence and stores them with size
B x 10 x 4096 where B is the batch size. 10 indicates the number of objects (i.e. 9) plus the full frame feature and 4096 is feature 
dimension. The RNN.py file has the model.

The dataset used is Street Accident (SA) dataset which contains 1266 videos for training and 467 videos for testing. It can be
downloaded from https://aliensunmin.github.io/project/dashcam/

The method anticipates accidents 3.76 seconds earlier on average. 

## Reference

```
@inproceedings{chan2016anticipating,
  title={Anticipating accidents in dashcam videos},
  author={Chan, Fu-Hsiang and Chen, Yu-Ting and Xiang, Yu and Sun, Min},
  booktitle={Asian Conference on Computer Vision},
  pages={136--153},
  year={2016},
  organization={Springer}
}
```

