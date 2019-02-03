# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.views.generic import FormView
from django.urls import reverse_lazy
from django.http import HttpResponse
from django.core.files.storage import get_storage_class
from django.utils import timezone
from django.utils.timezone import timedelta
from django.utils.http import urlencode

from streamimages.forms import UploadImageForm, DownloadImageForm
from streamimages.models import StreamImageModel
# Create your views here.


class FileDoesNotExist(Exception):
    def __init__(self, *args, **kwargs):
        self.fichier = kwargs.pop('file')
        super().__init__(*args, **kwargs)

def get_signed_url(fichier, expiration, methode, content_type):
    """
    Retourne un URL signé qui permet de modifier/lire le fichier
    Arguments:
        fichier (GoogleCloudFile):
        expiration (datetime.timedelta, int): durée de validité avant l'expiration du lien
        methode (str): méthode HTTP qui sera utilisé sur ce fichier
    """
    GoogleCloudStorage = get_storage_class()
    gcs = GoogleCloudStorage()
    print('content-type: {}'.format(content_type))
    blob = gcs.bucket.get_blob(fichier.name)
    if blob is None:
        raise FileDoesNotExist(file=fichier)
    return blob.generate_signed_url(
        expiration, method=methode, content_type=content_type)

def get_media_upload_url(media):
    return get_signed_url(
        media.frame, timezone.now() + timedelta(days=1), 'PUT', "image/jpeg")

def get_media_download_url(media):
    return '&'.join([get_signed_url(
        media.frame, timezone.now() + timedelta(days=1), 'GET', None),
        urlencode([('response-content-disposition', 'attachment;filename="{}"'.format(media.scene_id))])
    ])

class UploadImageView(FormView):
    form_class = UploadImageForm
    template_name = r"streaming_template.html"
    success_url = reverse_lazy("upload_image")

    def get_result_data(self, stream_image):
        return {
            'scene_id':  stream_image.scene_id,
            'upload_url': get_media_upload_url(stream_image)
        }

    def form_valid(self, form):
        form.save()
        data = self.get_result_data(form.instance)
        return HttpResponse(data, content_type="application/json")

class DownloadImageView(FormView):
    form_class = DownloadImageForm
    template_name = r"streaming_template.html"
    success_url = reverse_lazy("download_image")

    def get_result_data(self, image):
        dowload_url : ""
        return {
            'scene_id': image.scene_id,
            'download_url': dowload_url }

    def form_valid(self, form):
        StreamImageModel.objects.get(scene_id=self.request.POST["scene_id"])
        data = self.get_result_data(form.instance)
        return HttpResponse(data, content_type="application/json")