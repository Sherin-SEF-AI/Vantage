# Jetson (or any Linux) trigger program (Vantage). On the Jetson: sudo apt install libgpiod-dev (and pps-tools for
# --pps), then `make`. `make sim` builds the same program against a stand-in for the GPIO lines (no hardware).
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra
LIBS = -lpthread
SRC = trigger_jetson.c trigger_core.c

all: trigger_jetson
trigger_jetson: $(SRC) gpio_gpiod.c gpio.h trigger_core.h trigger_plan.h
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -I. $(SRC) gpio_gpiod.c -lgpiod $(LIBS) -o $@
trigger_jetson_pps: $(SRC) gpio_gpiod.c gpio.h trigger_core.h trigger_plan.h
	$(CC) $(CFLAGS) -DTB_HAVE_TIMEPPS -I. $(SRC) gpio_gpiod.c -lgpiod $(LIBS) -o $@
sim: trigger_jetson_sim
trigger_jetson_sim: $(SRC) gpio_sim.c gpio.h trigger_core.h trigger_plan.h
	$(CC) $(CFLAGS) -I. $(SRC) gpio_sim.c $(LIBS) -o $@
install: trigger_jetson
	install -m 0755 trigger_jetson /usr/local/bin/vantage-trigger
	install -m 0644 vantage-trigger.service /etc/systemd/system/vantage-trigger.service
include Makefile.host
clean:
	rm -f trigger_jetson trigger_jetson_pps trigger_jetson_sim host_sim
.PHONY: all sim install clean
