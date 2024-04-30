# Artificial Pancreas Medical Control System

This repository contains a three-part project focused on enhancing the Artificial Pancreas Medical Control System through data mining techniques, with a specific emphasis on the Medtronic 670G system.

## Table of Contents

- [Introduction](#introduction)
- [Project 1: Extracting Time Series Properties of Glucose Levels](#project-1)
- [Project 2: Machine Model Training for Meal Detection](#project-2)
- [Project 3: Cluster Validation for Carbohydrate Estimation](#project-3)

## Introduction

This project aims to improve the understanding and application of data mining techniques within the context of the Artificial Pancreas Medical Control System. The Medtronic 670G system comprises a continuous glucose monitor (CGM) and the Guardian Sensor, providing real-time blood glucose readings every five minutes. These readings are utilized by the MiniMed insulin pump to adjust insulin delivery.

## Project 1: Extracting Time Series Properties of Glucose Levels

In this phase, we extract key performance metrics from the sensor data, including:
- Percentage time in hyperglycemia (CGM > 180 mg/dL)
- Percentage of time in hyperglycemia critical (CGM > 250 mg/dL)
- Percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)
- Percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)
- Percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)
- Percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)

These metrics will be extracted for both Manual and Auto modes of operation.

## Project 2: Machine Model Training for Meal Detection

This segment involves training a machine learning model to discern between meal and no-meal time series data. The training dataset includes ground truth labels of Meal and No Meal for five subjects. Feature extraction techniques will be employed to ensure discriminative features are utilized for training the model.

## Project 3: Cluster Validation for Carbohydrate Estimation

Here, cluster validation techniques will be applied to estimate carbohydrate intake during meals. Ground truth values for each data point will be extracted, and clustering algorithms such as KMeans and DBSCAN will be employed to categorize meal data based on carbohydrate content.
