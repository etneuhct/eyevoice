from django import forms
from streamimages.models import StreamImageModel


class UploadImageForm(forms.ModelForm):

   class Meta:

        model = StreamImageModel
        fields = ['frame', 'time', 'scene_id']

class DownloadImageForm(forms.ModelForm):

   class Meta:

        model = StreamImageModel
        fields = ['scene_id']
