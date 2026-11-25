import time
from datetime import datetime

import board
import adafruit_dht
from smbus2 import SMBus
from gpiozero import DigitalInputDevice
from gpiozero.pins.lgpio import LGPIOFactory


INTERVAL = 5.0
ADS_ADDR = 0x48
MAX_ADDR = 0x57

# BCM GPIO numbering, not physical pin numbers.
DHT_PIN = board.D4       # Physical pin 7
DOOR_GPIO = 17           # Physical pin 11

# Change to False if your magnetic switch works in reverse.
DOOR_CLOSED_WHEN_CONTACT_CLOSED = True


def read_ecg_voltage(bus):
    # ADS1115:
    # A0 relative to GND, +/-4.096 V range,
    # single conversion, 128 samples/sec, comparator disabled.
    # Actual input must still remain between GND and VDD.
    config = 0xC383
    bus.write_i2c_block_data(
        ADS_ADDR, 0x01,
        [(config >> 8) & 0xFF, config & 0xFF]
    )

    deadline = time.monotonic() + 0.2

    while True:
        status = bus.read_i2c_block_data(ADS_ADDR, 0x01, 2)
        if status[0] & 0x80:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("ADS conversion timed out")
        time.sleep(0.002)

    data = bus.read_i2c_block_data(ADS_ADDR, 0x00, 2)
    raw = (data[0] << 8) | data[1]

    if raw & 0x8000:
        raw -= 65536

    volts = raw * 4.096 / 32768
    return raw, volts


def read_max_raw(bus):
    """Short optical sample burst; not a pulse/SpO2 algorithm."""
    part_id = bus.read_byte_data(MAX_ADDR, 0xFF)
    if part_id != 0x15:
        raise RuntimeError(
            f"Unexpected MAX part ID: 0x{part_id:02X}"
        )

    # Stop conversion while configuring and clearing FIFO.
    bus.write_byte_data(MAX_ADDR, 0x09, 0x80)

    try:
        # Disable interrupts; poll FIFO instead.
        bus.write_byte_data(MAX_ADDR, 0x02, 0x00)
        bus.write_byte_data(MAX_ADDR, 0x03, 0x00)

        # Clear FIFO pointers and overflow counter.
        for register in (0x04, 0x05, 0x06):
            bus.write_byte_data(MAX_ADDR, register, 0x00)

        # No averaging, FIFO rollover disabled.
        bus.write_byte_data(MAX_ADDR, 0x08, 0x00)

        # ADC range 4096 nA, 100 samples/s, 18-bit samples.
        bus.write_byte_data(MAX_ADDR, 0x0A, 0x27)

        # Low LED current for initial testing.
        bus.write_byte_data(MAX_ADDR, 0x0C, 0x1F)
        bus.write_byte_data(MAX_ADDR, 0x0D, 0x1F)

        # Red + IR acquisition mode.
        bus.write_byte_data(MAX_ADDR, 0x09, 0x03)

        deadline = time.monotonic() + 0.5

        while True:
            write_ptr = bus.read_byte_data(MAX_ADDR, 0x04) & 0x1F
            read_ptr = bus.read_byte_data(MAX_ADDR, 0x06) & 0x1F

            if write_ptr != read_ptr:
                break

            if time.monotonic() >= deadline:
                raise TimeoutError("MAX FIFO has no samples")

            time.sleep(0.01)

        data = bus.read_i2c_block_data(MAX_ADDR, 0x07, 6)

        red = (
            (data[0] << 16) | (data[1] << 8) | data[2]
        ) & 0x3FFFF

        ir = (
            (data[3] << 16) | (data[4] << 8) | data[5]
        ) & 0x3FFFF

        return red, ir

    finally:
        # Short bursts only: no continuous optical recording.
        bus.write_byte_data(MAX_ADDR, 0x09, 0x83)


def main():
    bus = None
    dht = None
    door = None
    factory = None

    try:
        try:
            bus = SMBus(1)
        except Exception as exc:
            print(f"I2C initialization error: {exc}")

        try:
            dht = adafruit_dht.DHT22(
                DHT_PIN, use_pulseio=False
            )
        except Exception as exc:
            print(f"DHT initialization error: {exc}")

        try:
            factory = LGPIOFactory()
            door = DigitalInputDevice(
                DOOR_GPIO,
                pull_up=True,
                bounce_time=0.05,
                pin_factory=factory,
            )
        except Exception as exc:
            print(f"Door initialization error: {exc}")

        print("WIRING TEST | Print interval: 5 seconds")
        print("ECG voltage and MAX raw signals only.")
        print("Press Ctrl+C to stop.")

        while True:
            cycle_start = time.monotonic()

            print("\n" + "=" * 54)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            if dht is not None:
                try:
                    temperature = dht.temperature
                    humidity = dht.humidity

                    if temperature is None or humidity is None:
                        raise RuntimeError("No complete DHT reading")

                    print(
                        f"DHT22 | Temperature: {temperature:.1f} C"
                        f" | Humidity: {humidity:.1f} %"
                    )
                except Exception as exc:
                    print(f"DHT22 | READ ERROR: {exc}")
            else:
                print("DHT22 | NOT INITIALIZED")

            if door is not None:
                try:
                    # Pull-up: pin LOW means switch connects to GND.
                    contact_closed = door.pin.state == 0

                    door_closed = (
                        contact_closed
                        if DOOR_CLOSED_WHEN_CONTACT_CLOSED
                        else not contact_closed
                    )

                    print(
                        f"DOOR  | {'CLOSED' if door_closed else 'OPEN'}"
                        f" | Contact: "
                        f"{'CLOSED' if contact_closed else 'OPEN'}"
                    )
                except Exception as exc:
                    print(f"DOOR  | READ ERROR: {exc}")
            else:
                print("DOOR  | NOT INITIALIZED")

            if bus is not None:
                try:
                    raw, voltage = read_ecg_voltage(bus)
                    print(
                        f"ECG   | A0: {voltage:.4f} V"
                        f" | ADC raw: {raw}"
                    )
                except Exception as exc:
                    print(f"ECG / ADS1115 | READ ERROR: {exc}")

                try:
                    red, ir = read_max_raw(bus)
                    print(f"MAX   | RED raw: {red} | IR raw: {ir}")
                    print("HR / SpO2 | Not calculated in wiring test")
                except Exception as exc:
                    print(f"MAX30102 | READ ERROR: {exc}")
            else:
                print("ECG / MAX | I2C NOT INITIALIZED")

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.0, INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\nTest stopped.")

    finally:
        for device in (dht, door, factory, bus):
            if device is not None:
                try:
                    if device is dht:
                        device.exit()
                    else:
                        device.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
