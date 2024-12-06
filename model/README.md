## Model Optimization and Deployment
In addition to extending the measurement modules, we have also optimized the model from an engineering perspective.  
The original model, which was packaged in the Torch format, has been converted into **ONNX** and **OpenVINO** formats.  
This conversion makes the model more lightweight and enables it to be deployed for inference on mobile devices, expanding its applicability and accessibility for real-world use cases.
 

Here we provide a compiled bin file that uses a small sample data set to train a minimal size model for demo presentation, which you can use directly