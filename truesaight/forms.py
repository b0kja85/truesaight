from django import forms

class VideoUploadForm(forms.Form):
    video_file = forms.FileField(
        label='Select a Video File',
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'id': 'videoFileInput',
            'accept': 'video/*'
        })
    )
