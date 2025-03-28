#include <stdio.h>
#include <unistd.h>
#include <wiringx.h>

// Define GPIO pins
#define A20 16 // LED pin
#define A18 22 // Switch pin

#define VERBOSE 1
#define SHOW_VALID_PINS 0

// Debounce delay (milliseconds)
#define DEBOUNCE_DELAY 50

// Function to read stable switch state
int readDebouncedInput(int pin)
{
    int firstRead, secondRead;
    do
    {
        firstRead = digitalRead(pin);
        usleep(DEBOUNCE_DELAY * 1000); // Wait 50ms
        secondRead = digitalRead(pin);
    } while (firstRead != secondRead); // Repeat if inconsistent
    return secondRead;
}

int main()
{
    int DUO_LED = A20;
    int DUO_SWITCH = A18;

    // Initialize WiringX
    if (wiringXSetup("milkv_duos", NULL) == -1)
    {
        printf("Failed to initialize WiringX.\n");
        wiringXGC();
        return 1;
    }
    printf("Initialize WiringX successfully.\n");

    // Validate GPIOs
    if (wiringXValidGPIO(DUO_LED) != 0 || wiringXValidGPIO(DUO_SWITCH) != 0)
    {
        printf("Invalid GPIO assignment.\n");
        return 1;
    }

    // Set pin modes
    pinMode(DUO_LED, PINMODE_OUTPUT);
    pinMode(DUO_SWITCH, PINMODE_INPUT); // Switch as input

    while (1)
    {
        int switch_state = readDebouncedInput(DUO_SWITCH);
        digitalWrite(DUO_LED, switch_state);

        if (VERBOSE)
            printf("Switch State: %d -> LED %s\n", switch_state, switch_state ? "ON" : "OFF");

        sleep(1); // Check every second
    }

    return 0;
}
