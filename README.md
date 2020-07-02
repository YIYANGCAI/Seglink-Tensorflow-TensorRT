# Convert Seglink Model to TensorRT from TensorFlow

## Requirements

TensorFlow-GPU 1.15.0

## Usage
Firstly, innstall the model from [Seglink](https://github.com/dengdan/seglink)
this model is for the input of 384 size.
Put the model in the ./ckpt file
then 
```
python ckpt2pb.py
```
to create the pb file
```
python get_res.py
```
to test the inference pf ckpt and pb model
