from django.db import models

class VideoProcessing(models.Model):
    """
    A model to represent a video processing task, storing information about the uploaded video,
    its processing status, the result of the analysis, and any relevant metadata.

    Attributes:
        video_file (FileField): The video file being processed.
        result (CharField): The result of the video analysis, either 'real' or 'fake'.
        confidence_score (FloatField): The confidence score indicating how confident the model is about the result.
        timestamp (DateTimeField): The timestamp when the video processing task was created.
        processed_frames_count (IntegerField): The number of frames processed during the task.
        status (CharField): The current status of the processing, with possible values:
                             'queued', 'processing', 'completed', or 'failed'.
        processing_duration (DurationField): The total time taken to process the video (optional, can be null).
        error_message (TextField): Any error message generated if the processing fails (optional, can be null).
    """

    # Video file to be processed
    video_file = models.FileField(upload_to='videos/')
    
    # Result of the processing task (e.g., 'real' or 'fake')
    result = models.CharField(max_length=10, choices=[('real', 'Real'), ('fake', 'Fake')])
    
    # Confidence score of the result
    confidence_score = models.FloatField()
    
    # Timestamp when the video was uploaded and processing began
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Number of frames processed during the task
    processed_frames_count = models.IntegerField()
    
    # Current status of the video processing task
    status = models.CharField(max_length=20, choices=[
        ('queued', 'Queued'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ])
    
    # Duration of the video processing 
    processing_duration = models.DurationField(null=True, blank=True)
    
    # Error message in case the processing fails 
    error_message = models.TextField(null=True, blank=True)


    def __str__(self):
        """
        Returns a string representation of the video processing task, 
        including the video name, processing reference number,
        processing result, and current status.

        Returns:
            str: String representation of the model instance.
        """
        return f" ID: {self.id} - Date: {self.timestamp.strftime('%Y-%m-%d')} File: {self.video_file.name}"
