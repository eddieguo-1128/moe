#!/bin/bash

# Export env variables
source .env.local

# Print 
echo $cuda_ver
echo $WORKING_DIR
echo $BASE_DIR
echo $DATA_DI
echo $CHECKPOINT_DIR
echo $OUTPUT_DIR
echo $TENSORBOARD_DIR

# Create
mkdir -p $BASE_DIR
mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $TENSORBOARD_DIR

