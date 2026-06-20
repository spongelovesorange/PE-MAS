from __future__ import annotations


def rated_output_current(power_w: float, output_voltage_v: float) -> float:
    if output_voltage_v == 0:
        raise ValueError("output_voltage_v must be non-zero")
    return power_w / output_voltage_v


def loss_at_efficiency(output_power_w: float, efficiency: float) -> float:
    if efficiency <= 0:
        raise ValueError("efficiency must be positive")
    return output_power_w / efficiency - output_power_w

