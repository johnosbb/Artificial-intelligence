#!/bin/sh

echo "auto.sh started at $(date)" >> /tmp/auto.log
echo "auto: waiting for usb0..." > /dev/kmsg

# Put the program you want to run automatically here
# To enable wlan, run this
# /usr/bin/start_wlan.sh


# Wait up to 10 seconds for usb0 to appear
COUNT=0
while ! ip link show usb0 >/dev/null 2>&1; do
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge 10 ]; then
        echo "auto:usb0 not found after 10 seconds" > /dev/kmsg
        exit 1
    fi
done


echo "auto:usb0 detected, proceeding..." > /dev/kmsg

# Now, we can set the IP address
echo "auto:setting IP adress of usb0 ..." > /dev/kmsg
ip link set usb0 up
ip addr flush dev usb0
ip addr add 192.168.42.250/24 dev usb0
echo "auto: Completed ..." > /dev/kmsg


