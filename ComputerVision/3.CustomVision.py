# !pip install azure-cognitiveservices-vision-customvision

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

ENDPOINT_Training = 'https://labuser58custom.cognitiveservices.azure.com/'
ENDPOINT_Prediction = 'https://labuser58custom-prediction.cognitiveservices.azure.com/'

training_key = '35555f8006bd4539b4eb17d0d288f4ee'
prediction_key = '0e86e64e5bac43989671021d90edba55'
prediction_resource_id = '/subscriptions/7ae06d59-97e1-4a36-bbfe-efb081b9b03b/resourceGroups/RG58/providers/Microsoft.CognitiveServices/accounts/labuser58custom'


credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT_Training, credentials)

print ("Creating project...")
project = trainer.create_project("Labuser58 Project")

Jajangmyeon_tag = trainer.create_tag(project.id, "Jajangmyeon")
Champon_tag = trainer.create_tag(project.id, "Champon")
Tangsuyug_tag = trainer.create_tag(project.id, "Tangsuyug")

print('Training...')
iteration = trainer.train_project(project.id)
while (iteration.status != 'Completed'):
  iteration = trainer.get_iteration(project.id, iteration.id)
  print('Training status' + iteration.status)

  time.sleep(10)

print('Done!')

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT_Prediction, prediction_credentials)

target_image_url = 'https://w.namu.la/s/d4c53737b61fec8cf0fa02206d85a5022fc5465593f2e0190648f7c5911acd836a5f7a1db0f19f0136ec1c178d782465a9455b31d178b79df5133fc6b493a41f712b0639f7b8a188e50189a15b74d987e49de963c1401191a1a03e6f80b96179'
result = predictor.classify_image_url(project.id, 'greatwall', target_image_url) #id, publish name, image

for prediction in result.predictions:
  print('\t' + prediction.tag_name + ': {0:.2f}%'.format(prediction.probability * 100))