import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


g = -9.81
k_air = 0.0035
vel_to_mph = 2.23694


def part_one() -> None:
    state = np.array([
        2000, 0
    ])

    eval_points = np.linspace(0, 100, 10000)
    
    def f(t:float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], g + k_air * state[1]**2])
    

    def height(t: float, state: np.ndarray) -> float:
        return state[0]
    height.terminal = True
    height.direction = -1
    

    solution = sp.integrate.solve_ivp(f, (0, 1e6), state, t_eval=eval_points, events=height)


    times = solution.t
    states = solution.y
    fall_time = solution.t_events[0][0]
    landing_speed = np.abs(solution.y_events[0][0, 1])
    print(f'Skydiver landed after {round(fall_time, 3)} seconds traveling at a speed of {round(landing_speed, 3)} m/s ({round(landing_speed * vel_to_mph, 3)} mph).')

    heights = states[0, :]
    velocities = states[1, :]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(times, heights)
    ax1.set_title("Position")

    ax2.plot(times, velocities)
    ax2.set_title("Velocity")

    plt.show()


def main() -> None:
    part_one()


if __name__ == "__main__":
    main()
