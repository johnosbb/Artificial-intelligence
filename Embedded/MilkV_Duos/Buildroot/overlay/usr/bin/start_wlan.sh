interface="wlan0"
max_attempts=100
attempt=0
log_file="/var/log/start_wlan.sh.log"

# Continuously attempt to detect if the interface exists, up to $max_attempts times
echo "Running start_wlan.sh" > "$log_file"
while [ $attempt -lt $max_attempts ]; do
    # Check if the wlan0 interface exists
    ip link show "$interface" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "$(date +'%Y-%m-%d %H:%M:%S') $interface interface exists, starting wpa_supplicant..." >> "$log_file"
        wpa_supplicant -B -i "$interfa e" -c /etc/wpa_supplicant.conf >> "$log_file"
        break  # Exit the loop if the interface is found
    else
        echo "$(date +'%Y-%m-%d %H:%M:%S') $interface interface not found, waiting..." >> "$log_file"
        sleep 1  # Wait for 1 second before checking again
        attempt=$((attempt + 1))  # Increment the attempt counter
    fi
done

# If the maximum number of attempts is reached and the interface still not found, output an error message
if [ $attempt -eq $max_attempts ]; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') Interface $interface not found after $max_attempts attempts" >> "$log_file"
fi


# Wait for wlan0 to be fully up (you can adjust the time based on your network's speed)
echo "$(date +'%Y-%m-%d %H:%M:%S') waiting before bringing the network up ..." >> "$log_file"
sleep 10

# Assign static IP
#echo "Assigning a static IP to wlan0" >> "$log_file"
#ifconfig wlan0 192.168.1.210 netmask 255.255.255.0 up

#echo "Static IP assigned to wlan0" >> "$log_file"
#sync

ifup wlan0

# Manually add my gateway to the routing table by specifying its mac address
arp -s 192.168.1.254 3c:9e:c7:1d:49:19
