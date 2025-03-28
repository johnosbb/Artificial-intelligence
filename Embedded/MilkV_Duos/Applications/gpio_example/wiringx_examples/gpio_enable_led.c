#include <stdio.h>
#include <unistd.h>
#include <wiringx.h>
#include <termios.h>

#define LED_PIN 16 // A20 (LED)

char get_keypress()
{
    struct termios oldt, newt;
    char ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

int main()
{
    if (wiringXSetup("milkv_duos", NULL) == -1)
    {
        printf("Failed to initialize WiringX.\n");
        return 1;
    }

    pinMode(LED_PIN, PINMODE_OUTPUT);
    printf("Press any key to turn on the LED...\n");
    get_keypress();

    // Turn LED ON
    digitalWrite(LED_PIN, HIGH);
    printf("LED is now ON!\n");

    while (1)
    {

        sleep(1);
    }

    return 0;
}
