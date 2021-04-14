# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2021-03-28
### Added
- a four-class classifier with bacteria pneumonia x-ray images
- model weights

## [1.0.1] - 2021-04-03
### Added
- notebook (classifier_1_0_1) used to output PR-curve, history of loss, sensitivity and specificiy

## [1.0.2] - 2021-04-04
### Added
- Randome sampling for training and testing sets
- Save statistics for PR curve
- Learning rate for new model

## [1.0.3] - 2021-04-11
### Changed
- Notebook change on saving the file for PR curve; the final image directory has to be removed in order to download the desired csv file.
### Added
- pr_curve_0410.csv corresponding to the model ran on 0410.

## [1.0.4] - 2021-04-12
### Changed
- Added data processing so that no patient leakage in the bacteria dataset
### Added
- pr_curve_0412_a_no_removal.csv pr curve data with no patient removal in the bacteria dataset
- pr_curve_0412_b_removal.csv pr curve data with patient removal in the bacteria dataset

## [1.0.5] - 2021-04-13
### Added
- pr_curve_0413_squeezenet10.csv pr curve on squeezenet
- pr_curve_0413_squeezenet10_b.csv without rotation 