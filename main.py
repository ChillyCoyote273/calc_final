import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from screeninfo import get_monitors


window_size = (8.0, 4.5)
for m in get_monitors():
    if m.is_primary:
        window_size = (m.width_mm / 25.4, m.height_mm / 25.4)

width = 0.5
euler_alpha = .9

g = -9.81
k_air = 0.0035
k_parachute = 0.175
vel_to_mph = 2.23694
parachute_height = 1000
openning_distance = 100
altitude_constant = 0.9999


class LinearInterpolation(sp.integrate.DenseOutput):
    def __init__(self, t_old, t, y_old, y):
        super().__init__(t_old, t)
        self.y_old = y_old
        self.y = y
    
    def _call_impl(self, t):
        if len(t.shape) > 0:
            return np.array([self.y_old + (i - self.t_min) / (self.t_max - self.t_min) * (self.y - self.y_old) for i in t])
        return self.y_old + (t - self.t_min) / (self.t_max - self.t_min) * (self.y - self.y_old)


class ForwardEuler(sp.integrate.OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, vectorized, support_complex=False, **extraneous):
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex)
        self.y_old = None
        self.h = 0.5

    def _step_impl(self) -> tuple[bool, None]:
        self.y_old = self.y.copy()
        derivative = self.fun(self.t, self.y)
        self.y = self.y + derivative * self.h
        self.t += self.h
        return (True, None)

    def _dense_output_impl(self):
        return LinearInterpolation(self.t_old, self.t, self.y_old, self.y)


def air_resistance(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], g + k_air * y[1]**2])

def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def smooth_step(x: float) -> float:
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    return 3 * x**2 - 2 * x**3

def smooth_air_resistance(h: float) -> float:
    fraction_openned = smooth_step((parachute_height - h) / openning_distance)
    return lerp(k_air, k_parachute, fraction_openned)

def air_resistance_parachute(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], g + smooth_air_resistance(y[0]) * y[1]**2])

def air_resistance_altitude(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], g + altitude_constant**max(y[0], 0) * smooth_air_resistance(y[0]) * y[1]**2])


part_one_conditions = (
    np.array([
        2.0e3, 0.0
    ]),
    air_resistance
)
part_two_conditions = (
    np.array([
        2.0e3, 0.0
    ]),
    air_resistance_parachute
)
part_three_conditions = (
    np.array([
        4.0e4, 0.0
    ]),
    air_resistance_altitude
)


def simulate(conditions) -> None:
    eval_points = np.arange(0, 1000, 0.5)

    def height(t: float, y: np.ndarray) -> float:
        return y[0]
    height.terminal = True
    height.direction = -1

    def acceleration(t: float, y: np.ndarray) -> float:
        return conditions[1](t, y)[1]

    solution = sp.integrate.solve_ivp(conditions[1], (0, 1e6), conditions[0], method='RK23', t_eval=eval_points, events=[height, acceleration])
    solution_euler = sp.integrate.solve_ivp(conditions[1], (0, 1e6), conditions[0], method=ForwardEuler, events=[height, acceleration])

    print(f'Dormand Prince method took {solution.nfev} physics calculations.')
    times = solution.t
    states = solution.y
    fall_time = solution.t_events[0][0]
    landing_speed = np.abs(solution.y_events[0][0, 1])
    print(f'Dormand Prince: Skydiver landed after {round(fall_time, 3)} seconds traveling at a speed of {round(landing_speed, 3)} m/s ({round(landing_speed * vel_to_mph, 3)} mph).')
    if len(solution.t_events[1]) != 0:
        max_vel_time = solution.t_events[1][0]
        max_vel_height = solution.y_events[1][0, 0]
        max_vel = np.abs(solution.y_events[1][0, 1])
        print(f'Dormand Prince: At {round(max_vel_time, 3)} seconds the skydiver was falling fastest at an altitude of {round(max_vel_height * 1e-3, 3)} kilometers and a speed of {round(max_vel, 3)} m/s ({round(max_vel * vel_to_mph, 3)} mph).')
    
    print(f'Euler method took {solution_euler.nfev} physics calculations.')
    euler_times = solution_euler.t
    euler_states = solution_euler.y
    euler_fall_time = solution_euler.t_events[0][0]
    euler_landing_speed = np.abs(solution_euler.y_events[0][0, 1])
    print(f'Euler: Skydiver landed after {round(euler_fall_time, 3)} seconds traveling at a speed of {round(euler_landing_speed, 3)} m/s ({round(euler_landing_speed * vel_to_mph, 3)} mph).')
    if len(solution_euler.t_events[1]) != 0:
        euler_max_vel_time = solution_euler.t_events[1][0]
        euler_max_vel_height = solution_euler.y_events[1][0, 0]
        euler_max_vel = np.abs(solution_euler.y_events[1][0, 1])
        print(f'Euler: At {round(euler_max_vel_time, 3)} seconds the skydiver was falling fastest at an altitude of {round(euler_max_vel_height * 1e-3, 3)} kilometers and a speed of {round(euler_max_vel, 3)} m/s ({round(euler_max_vel * vel_to_mph, 3)} mph).')

    heights = states[0, :]
    velocities = states[1, :]

    euler_heights = euler_states[0, :]
    euler_velocities = euler_states[1, :]


    _, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=window_size)

    ax1.plot(times, heights, c='r', lw=width)
    ax1.plot(euler_times, euler_heights, alpha=euler_alpha, c='b', lw=width)
    ax1.set_title('Position')

    ax2.plot(times, velocities, c='r', lw=width)
    ax2.plot(euler_times, euler_velocities, alpha=euler_alpha, c='b', lw=width)
    ax2.set_title('Velocity')

    plt.show()


def part_one() -> None:
    print('\nPart 1:')
    simulate(part_one_conditions)

def part_two() -> None:
    print('\nPart 2:')
    simulate(part_two_conditions)

def part_three() -> None:
    print('\nPart 3:')
    simulate(part_three_conditions)

def part_four() -> None:
    print('\nPart 4:')
    print(f'Terminal velocity without a parachute is {round(np.sqrt(np.abs(g) / k_air), 3)} m/s ({round(vel_to_mph * np.sqrt(np.abs(g) / k_air), 3)} mph).')
    print(f'Terminal velocity with a parachute is {round(np.sqrt(np.abs(g) / k_parachute), 3)} m/s ({round(vel_to_mph * np.sqrt(np.abs(g) / k_parachute), 3)} mph).')

    mass = 2000
    desired_speed = 5.0 / vel_to_mph
    print(f'To effectively slow a Jeep Wrangler weighing {mass} kg ({int(mass * 2.2)} lbs) down to a speed of 5 mph, you would need a parachute with {round(-g * mass / (0.37 * desired_speed**2), 2)} square meters of surface area.', end=' ')
    print(f'It could be a square with {round(np.sqrt(-g * mass / (0.37 * desired_speed**2)))} meter long sides.')


def main() -> None:
    part_one()
    part_two()
    part_three()
    part_four()


if __name__ == '__main__':
    main()
