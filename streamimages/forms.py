from django import forms
from streamimages.models import StreamImageModel


class StreamForm(forms.ModelForm):

   class Meta:

        model = StreamImageModel
        fields = ['frame', 'time', 'scene_id']
