# STM32F411 trigger box (Vantage). Needs arm-none-eabi-gcc (Arm GNU Toolchain) and the CMSIS headers of
# STM32CubeF4 (https://github.com/STMicroelectronics/STM32CubeF4, or cmsis_device_f4 + CMSIS_5's Core/Include).
CUBE ?= $(HOME)/STM32CubeF4
CMSIS_DEVICE ?= $(CUBE)/Drivers/CMSIS/Device/ST/STM32F4xx/Include
CMSIS_CORE ?= $(CUBE)/Drivers/CMSIS/Include
CROSS ?= arm-none-eabi-
CC = $(CROSS)gcc
OBJCOPY = $(CROSS)objcopy
CFLAGS = -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -O2 -std=c11 -Wall -Wextra -ffunction-sections -fdata-sections \
         -DSTM32F411xE -I. -I$(CMSIS_DEVICE) -I$(CMSIS_CORE)
LDFLAGS = -T stm32f411.ld -nostartfiles --specs=nano.specs -Wl,--gc-sections
SRC = main.c startup.c trigger_core.c

all: build/trigger.bin
build/trigger.elf: $(SRC) trigger_core.h trigger_plan.h stm32f411.ld
	mkdir -p build
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $@
build/trigger.bin: build/trigger.elf
	$(OBJCOPY) -O binary $< $@
flash: build/trigger.bin
	st-flash --reset write build/trigger.bin 0x08000000
flash-openocd: build/trigger.elf
	openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "program build/trigger.elf verify reset exit"
include Makefile.host
clean:
	rm -rf build host_sim
.PHONY: all flash flash-openocd clean
