# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.

class StreamImageModel(models.Model):

    frame = models.ImageField(verbose_name='img', upload_to=r'streamimages/images', null=True, blank=True)
    time = models.TimeField()
    scene_id = models.IntegerField(unique=True)
    uri = models.URLField(blank=True, null=True)
