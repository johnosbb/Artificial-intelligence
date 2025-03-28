#include <stdio.h>
#include <unistd.h>

#include <wiringx.h>

// ## blink

// This example demonstrates how to read a switch connected to a GPIO pin.
// It uses the WiringX library to read the GPIO pin's voltage level.
// The code includes platform initialization and GPIO manipulation methods from the WiringX library.

#define VERBOSE 0
#define SHOW_VALID_PINS 0

#define A18 22 // A18 is on pin 22 of J3 [498 of XGPIOA]

int main()
{
    int DUO_SWITCH = A18; // Use A18 (Pin 22) as input

    // Initialize WiringX for Milk-V DuoS
    if (wiringXSetup("milkv_duos", NULL) == -1)
    {
        printf("Failed to initialize WiringX.\n");
        wiringXGC();
        return 1;
    }
    printf("Initialize WiringX successfully.\n");

    // Show valid pins if enabled
    if (SHOW_VALID_PINS)
    {
        for (int i = 0; i < 100; i++)
        {
            int is_valid = wiringXValidGPIO(i);
            if (is_valid)
                printf("GPIO %d is Valid: %d\n", i);
        }
    }

    // Validate GPIO
    if (wiringXValidGPIO(DUO_SWITCH) != 0)
    {
        printf("Invalid GPIO %d\n", DUO_SWITCH);
        return 1;
    }

    // Set pin mode to INPUT
    pinMode(DUO_SWITCH, PINMODE_INPUT);

    // Read and print input value in a loop
    while (1)
    {
        int state = digitalRead(DUO_SWITCH);
        printf("GPIO %d Input State: %d\n", DUO_SWITCH, state);
        sleep(1); // Read every second
    }

    return 0;
}