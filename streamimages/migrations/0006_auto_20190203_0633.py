# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2019-02-03 11:33
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('streamimages', '0005_streamimagemodel_uri'),
    ]

    operations = [
        migrations.AlterField(
            model_name='streamimagemodel',
            name='scene_id',
            field=models.IntegerField(unique=True),
        ),
        migrations.AlterField(
            model_name='streamimagemodel',
            name='time',
            field=models.TimeField(),
        ),
    ]
