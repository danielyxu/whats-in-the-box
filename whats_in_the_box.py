import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification, ASTForAudioClassification, Pipeline
import torch
import librosa
from datasets import load_dataset


# Housekeeping functions
def load_audio(file_path, sr = None):
    audio_array, sampling_rate = librosa.load(file_path, sr=sr)
    return audio_array, sampling_rate

# Model and pipeline functions
def load_model_and_extractor(model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, extractor

def load_pipeline(task: str = "zero-shot-audio-classification",
                  model: str = "laion/clap-htsat-unfused") -> Pipeline:
    return pipeline(task=task,
                    model=model)

# model="laion/larger_clap_music",
# model = "laion/clap-htsat-fused",
# model = "laion/clap-htsat-unfused",
# model = 'laion/larger_clap_general'
# model = 'ClapCap'

def load_clap_classifier() -> Pipeline:
    return load_pipeline(task="zero-shot-audio-classification",
                         model="laion/clap-htsat-unfused")

def load_ast_classifier() -> Pipeline:
    return load_pipeline(task="audio-classification",
                         model="MIT/ast-finetuned-audioset-10-10-0.4593")

# Audio classification functions
MATERIAL_LABELS = ["wood", "metal", "glass", "plastic", "ceramic", "cardboard", "foam"]
UNSURE_LABELS = ["others"]

def enumerate_collision_labels(labels):
    collision_labels = []
    for index, label_1 in enumerate(labels):
        for label_2 in labels[index+1:]:
            collision_labels.append(f"{label_1} colliding with {label_2}")
    return collision_labels



# Audio segmentation functions


#Box functions
class Box:
    def __init__(self, 
                 audio_file_path, 
                 object_labels):
        self.audio_file_path = audio_file_path
        self.audio_array, self.sampling_rate = load_audio(audio_file_path)
        self.audio_duration = librosa.get_duration(y=self.audio_array, sr=self.sampling_rate)
        self.object_labels = object_labels
        self.collision_labels = enumerate_collision_labels(object_labels)
        self.candidate_labels = self.object_labels + self.collision_labels + MATERIAL_LABELS + UNSURE_LABELS

    def get_candidate_scores(self,
                                classifier: pipeline = None,
                                verbose: bool = True,
                                override: bool = False):
            if hasattr(self, "candidate_scores") and not override:
                if verbose:
                    print("Candidate scores already computed, returning cached values")
                return self.candidate_scores
    
            if classifier is None:
                classifier = load_clap_classifier()
    
            candidate_scores = classifier(self.audio_array, self.candidate_labels)
    
            self.candidate_scores = candidate_scores
    
            if verbose:
                print(f"Audio file: {self.audio_file_path}")
                print("Candidate label scores:")
                for label, score in candidate_scores.items():
                    print(f"{label}: {score:.3f}")
            return candidate_scores

class Boxes:
    """
    Class to represent the boxes in the audio file
    """
    def __init__(self,
                 audio_file_path,
                 num_boxes: int,
                 object_labels: list,
                 ):
        self.audio_file_path = audio_file_path
        self.audio_array, self.sampling_rate = load_audio(audio_file_path)
        self.audio_duration = librosa.get_duration(y=self.audio_array, sr=self.sampling_rate)

        self.num_boxes = num_boxes
        self.object_labels = object_labels
        self.collision_labels = enumerate_collision_labels(object_labels)
        self.candidate_labels = self.object_labels + self.collision_labels + MATERIAL_LABELS + UNSURE_LABELS

    def get_candidate_scores(self,
                             classifier: pipeline = None,
                             verbose: bool = True,
                             override: bool = False):
        if hasattr(self, "candidate_scores") and not override:
            if verbose:
                print("Candidate scores already computed, returning cached values")
            return self.candidate_scores

        if classifier is None:
            classifier = load_clap_classifier()

        candidate_scores = classifier(self.audio_array, self.candidate_labels)

        self.candidate_scores = candidate_scores

        if verbose:
            print(f"Audio file: {self.audio_file_path}")
            print(f"Number of boxes: {self.num_boxes}")
            print("Candidate label scores:")
            for label, score in candidate_scores.items():
                print(f"{label}: {score:.3f}")
        return candidate_scores

if __name__ == "__main__":
    print ("hello world")
