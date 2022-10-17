# !pip install azure-cognitiveservices-vision-customvision

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

ENDPOINT = 'https://labuser58custom.cognitiveservices.azure.com/'

training_key = '35555f8006bd4539b4eb17d0d288f4ee'
prediction_key = '0e86e64e5bac43989671021d90edba55'
prediction_resource_id = '/subscriptions/7ae06d59-97e1-4a36-bbfe-efb081b9b03b/resourceGroups/RG58/providers/Microsoft.CognitiveServices/accounts/labuser58custom'

publish_iteration_name = 'classifyModel'

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

print ("Creating project...")
project = trainer.create_project("Labuser58 Project")

Jajangmyeon_tag = trainer.create_tag(project.id, "Jajangmyeon")
Champon_tag = trainer.create_tag(project.id, "Champon")
Tangsuyug_tag = trainer.create_tag(project.id, "Tangsuyug")

import time

print('Training...')
iteration = trainer.train_project(project.id)
while (iteration.status != 'Completed'):
  iteration = trainer.get_iteration(project.id, iteration.id)
  print('Training status' + iteration.status)

  time.sleep(2)

print('Done!')