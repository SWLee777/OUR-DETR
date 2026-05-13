from ultralytics import RTDETR

# Load a model
model = RTDETR("ultralytics\\cfg\\models\\rt-detr\\OUR-DETR.yaml")
# Use the model
model.train(data="ultralytics\\cfg\\datasets\\RS-TOD.yaml",
            cfg="ultralytics\\cfg\\default.yaml", epochs=150)  # train the model

