from django import forms
from streamimages.models import StreamImageModel


class UploadImageForm(forms.ModelForm):

   class Meta:

        model = StreamImageModel
        fields = ['frame', 'time', 'scene_id']

class DownloadImageForm(forms.Form):

   scene_id = forms.IntegerField()