#!/bin/sh
# Example Pins


BUILTIN_LED_PIN=509
LED_GPIO=/sys/class/gpio/gpio${BUILTIN_LED_PIN}
if test -d ${LED_GPIO}; then
    echo "PIN ${BUILTIN_LED_PIN} already exported"
else
    echo ${BUILTIN_LED_PIN} > /sys/class/gpio/export
fi
echo out > ${LED_GPIO}/direction
echo 1 > ${LED_GPIO}/value
while true; do
    
    sleep 0.5
done