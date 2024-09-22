#!/bin/bash

# Specify the path to the SD card image
IMAGE_FILE="sdcard.img"

# Function to display usage
usage() {
    echo "Usage: $0 /dev/sdX"
    echo "Where /dev/sdX is the device path of the SD card (e.g., /dev/sdb)"
    exit 1
}

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    usage
fi

# Get the SD card device
SD_CARD="$1"

# Check if the SD card device exists
if [ ! -b "$SD_CARD" ]; then
    echo "Error: $SD_CARD does not exist or is not a block device."
    exit 1
fi

# Check the size of the SD card
SD_CARD_SIZE=$(lsblk -b -n -o SIZE "$SD_CARD")
IMAGE_SIZE=$(stat -c %s "$IMAGE_FILE")

# Convert sizes to human-readable format
SD_CARD_SIZE_HR=$(numfmt --to=iec-i --suffix=B "$SD_CARD_SIZE")
IMAGE_SIZE_HR=$(numfmt --to=iec-i --suffix=B "$IMAGE_SIZE")

# Display sizes
echo "SD Card Size: $SD_CARD_SIZE_HR"
echo "Image Size: $IMAGE_SIZE_HR"

# Check if the SD card is large enough
if [ "$SD_CARD_SIZE" -lt "$IMAGE_SIZE" ]; then
    echo "Error: SD card size ($SD_CARD_SIZE bytes) is smaller than the image size ($IMAGE_SIZE bytes)."
    exit 1
fi

# Confirm with the user before proceeding
echo "This will write $IMAGE_FILE to $SD_CARD. All data on the card will be lost!"
read -p "Are you sure you want to continue? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" ]]; then
    echo "Operation canceled."
    exit 0
fi

# Unmount the SD card if it is mounted
umount "$SD_CARD"* 2>/dev/null

# Write the image to the SD card
echo "Writing $IMAGE_FILE to $SD_CARD..."
sudo dd if="$IMAGE_FILE" of="$SD_CARD" bs=4M status=progress

# Sync to ensure all data is written
sync

echo "Done! $IMAGE_FILE has been written to $SD_CARD."
