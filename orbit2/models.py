from django.db import models
from django.contrib.auth.models import User
import uuid
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


class Pendency(models.Model):
    vertical = models.CharField(max_length=255 , null = True , blank=True)
    description = models.CharField(max_length=255, null=True, blank=True)
    color = models.CharField(max_length=50, null=True, blank=True)
    brand = models.CharField(max_length=50, blank=True , null=True)
    keywords = models.CharField(max_length=255, blank=True , null=True)  
    price = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    image = models.ImageField(upload_to='product_images/', blank=True, null=True)
    pid = models.CharField(max_length=255)
    itm_id = models.CharField(max_length=255)
    tid = models.CharField(max_length=255)
    features = models.BinaryField(null=True , blank=True)
    related_pids = models.ManyToManyField("RelatedPids")


    def save(self, *args, **kwargs):
        if self.image:
            features = self.extract_features(self.image)
            if features is not None:
                self.features = features.tobytes()
        self.name = self.tid
        super().save(*args, **kwargs)


    def extract_features(self, image_file):
        model = VGG16(weights='imagenet', include_top=False)
        try:
            img = Image.open(image_file)
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            features = model.predict(img_array)
            features = np.reshape(features, (7*7*512,))
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def __str__(self):
        return self.tid


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    casper_id = models.CharField(max_length=8)
    image = models.ImageField(default="default.jpg" , upload_to="profile_pics" )

    def __str__(self):
        return f"{self.user.username} Profile"
    

class physicalOrphan(models.Model):
    oid = models.CharField(max_length=255)
    tid = models.OneToOneField(Pendency , on_delete=models.CASCADE , default=None , blank=True)
    image = models.ImageField(upload_to='orphan_images/', blank=True, null=True)
    def __str__(self):
        return self.tid

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    query = models.CharField(max_length=255)
    filename = models.CharField(max_length=255 ,  blank=True, null=True)
    uuid = models.CharField(editable=False, unique=True , max_length=255 , blank=True , null=True)
    reconciled = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.query}"

    class Meta:
        ordering = ['-timestamp']

class ReconciliationRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    tid = models.CharField(max_length=255)
    reconciled_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} reconciled TID {self.tid} at {self.reconciled_at}"


class RelatedPids(models.Model):
    pid = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.pid
