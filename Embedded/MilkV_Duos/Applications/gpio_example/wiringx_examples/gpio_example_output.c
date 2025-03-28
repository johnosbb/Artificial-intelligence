#include <stdio.h>
#include <unistd.h>

#include <wiringx.h>

// ## blink

// This example demonstrates how to control an LED connected to a GPIO pin.
// It uses the WiringX library to toggle the GPIO pin's voltage level, resulting in the LED blinking.
// The `blink.c` code includes platform initialization and GPIO manipulation methods from the WiringX library.
// 20 Pin A20 - SG2000 Num 500

/*
     3.3V MCU (ESP32, DuoS, etc.)
     ┌──────────────────────────┐
     │ A20   ───┬─────────┐      │
     │       ───┤         │      │
     │       ───┤ ULN2803 │      │
     │       ───┤        OUTPUTS ├───(−) LED1 --330 Omh resistor --- 5V
     │       ───┤         │      ├───(−)
     │       ───┤         │      ├───(−)
     │       ───┤         │      ├───(−)
     │          └─────────┘      │
     │               │           │
     │              GND          │
     └──────────────┴────────────┘
                     │
           ┌────────┴────────┐
           │  5V or 12V Load │
           │   Power Supply  │
           └────────────────┘


*/
#define A20 20 // A20 [XGIOA 500] is on pin 20 of J3
#define VERBOSE 0
#define SHOW_VALID_PINS 1

int main()
{

    int DUO_LED = A20;

    // DuoS:    milkv_duos
    if (wiringXSetup("milkv_duos", NULL) == -1)
    {
        printf("Failed to initialize WiringX.\n");
        wiringXGC();
        return 1;
    }
    else
        printf("Initialize WiringX successfully.\n");
    if (SHOW_VALID_PINS)
    {
        for (int i = 0; i < 100; i++)
        {
            int is_valid = wiringXValidGPIO(i);
            if (is_valid)
                printf("GPIO %d is Valid: %d\n", i);
        }
    }
    if (wiringXValidGPIO(DUO_LED) != 0)
    {
        printf("Invalid GPIO %d\n", DUO_LED);
        return 1;
    }

    pinMode(DUO_LED, PINMODE_OUTPUT);

    while (1)
    {
        if (VERBOSE)
            printf("Duo LED GPIO (wiringX) %d: High\n", DUO_LED);
        digitalWrite(DUO_LED, HIGH);
        sleep(1);
        if (VERBOSE)
            printf("Duo LED GPIO (wiringX) %d: Low\n", DUO_LED);
        digitalWrite(DUO_LED, LOW);
        sleep(1);
    }

    return 0;
}
