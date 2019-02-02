# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.views.generic import FormView
from streamimages.forms import StreamForm
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect
from streamimages.models import StreamImageModel
# Create your views here.

class StreamView(FormView):
    form_class = StreamForm
    template_name = r"streaming_template.html"
    success_url = reverse_lazy("streaming")

    def form_valid(self, form):
        form.save()
        return HttpResponseRedirect(self.get_success_url())