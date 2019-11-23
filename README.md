# ObjTracker
This project contains flask-based deeplab v3 server that receives images as json format and semantic segmantation extraction client code that sends and receives segmentation result via server.

# Description
## Dataset
Dataset contains actual ADE20k validation dataset for livingroom as well as ground truth segmentation images and estimated segmentation images. The model itself has been trained using ADE20k training dataset. At the moment, the categories for segmentation is: "door", "sofa", "shelf", "table."

## Deeplab API
This folder contains both server and segmentation client. For server, there is "flaskServer.py" that uses Flask to setup server. To start server: 
> python FlaskServer.py bdist_wheel

To test semantic segmentation, use JK Test.ipynb on jupyter notebook. At the moment, I've setup jupyter such that anyone can access the code via:
> http://143.248.96.69:8888/

