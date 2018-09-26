#!bin/bash

echo "Getting the X2 downscaled TRAINING data"
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
echo "Done!"

echo "Getting the X2 downscaled VALIDATION data"
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
echo "Done!"

echo "Getting the X4 downscaled TRAINING data"
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
echo "Done!"

echo "Getting the X4 downscaled VALIDATION data"
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
echo "Done!"

echo "Getting TRAINING data that are not downscaled"
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
echo "Done!"

echo "Getting VALIDATION data that are not downscaled"
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
echo "Done!"
