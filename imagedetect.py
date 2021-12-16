from imageai.Detection import ObjectDetection
import os

detector = ObjectDetection()

model_path = "./models/yolo-tiny.h5"


path =".\input"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()



##detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)


image_index = 0 
for image in os.listdir(path):
    image_index+=1
    current_input_path = os.path.join(path, image)
    current_output_path = os.path.join("./output/" + str(image_index)+ ".png")
    custom = detector.CustomObjects(person=True, bicycle=True)
    detections = detector.detectCustomObjectsFromImage(
                        custom_objects = custom,
                        input_image = current_input_path, 
                        output_image_path = current_output_path)
    for eachItem in detections:
        print(eachItem["name"] , " : ", eachItem["percentage_probability"])