/* startup.c: reset handler and vector table for the STM32F411 (RM0383 table 37: 86 peripheral interrupts).
 * Only the handlers the trigger box uses are named; every other interrupt stops in Default_Handler. */
#include <stdint.h>

extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;
extern int main(void);

uint32_t SystemCoreClock = 16000000u;            /* the HSI until main() switches to the PLL */

void Reset_Handler(void);
void Default_Handler(void);
void TIM2_IRQHandler(void);
void TIM5_IRQHandler(void);
void SysTick_Handler(void);

void Reset_Handler(void)
{
    uint32_t *s = &_sidata, *d = &_sdata;
    while (d < &_edata)
        *d++ = *s++;
    for (d = &_sbss; d < &_ebss;)
        *d++ = 0;
    main();
    for (;;) {
    }
}

void Default_Handler(void)
{
    for (;;) {
    }
}

#define N_VECTORS (16 + 86)
#define IRQ(n) (16 + (n))

__attribute__((section(".isr_vector"), used))
void (*const vectors[N_VECTORS])(void) = {
    [0] = (void (*)(void))(&_estack),
    [1] = Reset_Handler,
    [2 ... 14] = Default_Handler,
    [15] = SysTick_Handler,
    [16 ... IRQ(28) - 1] = Default_Handler,
    [IRQ(28)] = TIM2_IRQHandler,                  /* TIM2 global interrupt */
    [IRQ(28) + 1 ... IRQ(50) - 1] = Default_Handler,
    [IRQ(50)] = TIM5_IRQHandler,                  /* TIM5 global interrupt */
    [IRQ(50) + 1 ... N_VECTORS - 1] = Default_Handler,
};
