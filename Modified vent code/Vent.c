#include <msp430.h> 
// Configure pins (all in P1)
#define ON_LED BIT2
#define RESISTOR BIT5
#define XBEE3  BIT6
#define SERVO  BIT7

// Servo status and parameters
#define CLOSED  0
#define OPEN 1
#define FALSE 0
#define TRUE 1
#define SERVO_ANGLE_BIG 1000
#define SERVO_ANGLE_SMALL 1000
#define TOTAL_INTERVAL 20
#define TOTAL_DURATION 180

// Time open definitions (based on pulse duration)
#define TIME_OPEN_25_PERCENT 5     // 25% of 20 seconds = 5 seconds
#define TIME_OPEN_50_PERCENT 10    // 50% of 20 seconds = 10 seconds
#define TIME_OPEN_100_PERCENT 20   // 100% of 20 seconds = 20 seconds

// Function declarations
void initTimer_A(void);
void delay_seconds(int seconds);
void control_valve(int time_open);
void adjustTimeOpen(void);

// Global Variables
unsigned char servo_status;
unsigned int temp;
unsigned long overflow_counter;
unsigned int pulse_duration;    // Stores the duration of the high pulse
unsigned int time_open;         // Time to keep valve open

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;   // Stop watchdog timer
    servo_status = CLOSED;

    // Configure output pins
    P1DIR |= ON_LED; // P1.2 LED enable output
    P1OUT |= ON_LED; // LED on
    P1DIR |= SERVO; // P1.7 enable output for servo
    P1OUT &= ~(SERVO); // Start off
    P1DIR |= RESISTOR; // P1.5 RESISTOR enable output
    P1OUT |= (RESISTOR); // Start on

    // Configure input pin for XBee
    P1DIR &= ~(XBEE3); // P1.6 (XBEE3) as input
    P1IES &= ~(XBEE3); // Look for low to high edge initially
    P1IFG &= ~(XBEE3); // Clear interrupt flag
    P1IE |= XBEE3;     // Enable interrupt for P1.6

    // MCLK = SMCLK = 1MHz
    DCOCTL = 0;
    BCSCTL1 = CALBC1_1MHZ;
    DCOCTL = CALDCO_1MHZ;

    initTimer_A();
    __enable_interrupt();

    while (TRUE) {
        // Control the valve for 3 minutes based on the received time_open value
        for (int i = 0; i < (TOTAL_DURATION / TOTAL_INTERVAL); i++) {
            control_valve(time_open);
        }
    }
}

// Function to control the valve based on the given time_open
void control_valve(int time_open) {
    int time_close = (TOTAL_INTERVAL - time_open) / 2;

    // Close for time_close seconds
    servo_status = CLOSED;
    P1OUT &= ~(SERVO);
    delay_seconds(time_close);

    // Open for time_open seconds
    servo_status = OPEN;
    P1OUT |= (SERVO);
    delay_seconds(time_open);

    // Close for time_close seconds
    servo_status = CLOSED;
    P1OUT &= ~(SERVO);
    delay_seconds(time_close);
}

// Function to initialize Timer A
void initTimer_A(void) {
    // Timer0_A3 Config
    TACCTL0 |= CCIE; // Enable CCR0 interrupt
    TACCTL1 |= CCIE; // Enable CCR1 interrupt
    TACCR1 = 20000;  // CCR1 initial value (Period)
    TACCR0 = TACCR1 + SERVO_ANGLE_SMALL + (SERVO_ANGLE_BIG * servo_status); // CCR0 initial value (Pulse Width)
    TACTL = TASSEL_2 + ID_0 + MC_2; // Use SMCLK, SMCLK/1, Counting Continuous Mode
}

// Delay function for seconds
void delay_seconds(int seconds) {
    unsigned long desired_overflows = overflow_counter + 15 * seconds; // Calculated delay
    while (overflow_counter < desired_overflows); // Wait until desired delay is reached
}

// Timer ISR for CCR0
#pragma vector = TIMER0_A0_VECTOR
__interrupt void Timer_A_CCR0_ISR(void) {
    P1OUT &= ~(SERVO); // Set SERVO low
    TACCR0 = TACCR1 + SERVO_ANGLE_SMALL + (SERVO_ANGLE_BIG * servo_status); // Add one Period
}

// Timer ISR for CCR1 (Servo control)
#pragma vector = TIMER0_A1_VECTOR
__interrupt void Timer_A_CCR1_ISR(void) {
    temp = TAIV;
    if (temp == TAIV_TACCR1) {
        P1OUT |= (SERVO); // Set SERVO High
        TACCR1 += 20000;  // Add one Period
    }
}

// Port 1 ISR for edge detection (pulse measurement from XBee)
#pragma vector = PORT1_VECTOR
__interrupt void Servo_change(void) {
    if ((P1IES & XBEE3) == 0) {  // Rising edge detected
        P1IES |= XBEE3;          // Now look for falling edge
        TACTL |= TACLR;          // Clear TimerA count
        pulse_duration = 0;      // Reset pulse duration
    } else {                     // Falling edge detected
        P1IES &= ~XBEE3;         // Now look for rising edge
        pulse_duration = TAR;    // Store the pulse duration from TimerA
        adjustTimeOpen();        // Adjust time_open based on pulse_duration
    }
    P1IFG &= ~XBEE3;             // Clear interrupt flag
}

// Function to adjust time_open based on measured pulse duration
void adjustTimeOpen(void) {
    if (pulse_duration < 500) {  // Short pulse corresponds to '1'
        time_open = TIME_OPEN_25_PERCENT;
    } else if (pulse_duration < 1500) {  // Medium pulse corresponds to '2'
        time_open = TIME_OPEN_50_PERCENT;
    } else {  // Long pulse corresponds to '3'
        time_open = TIME_OPEN_100_PERCENT;
    }
}
