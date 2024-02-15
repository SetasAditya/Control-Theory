import numpy as np

def piecewise_linear_fit(data, N):
    data = sorted(data, key=lambda x: x[0])  # Sort the data points in ascending order of x

    num_intervals = min(N, len(data) - 1)
    interval_size = len(data) // num_intervals

    intervals = np.zeros((num_intervals, 4))

    start_idx = 0
    end_idx = interval_size

    for i in range(num_intervals):
        x_interval = [x for x, _ in data[start_idx:end_idx]]
        y_interval = [y for _, y in data[start_idx:end_idx]]

        X = np.array([x_interval, np.ones(len(x_interval))]).T
        m, c = np.linalg.lstsq(X, y_interval, rcond=None)[0]

        if i > 0:
            intervals[i-1, 2] = np.arctan(m_prev)
            intervals[i-1, 3] = np.arctan(m)

        m_prev = m
        intervals[i, 0] = m
        intervals[i, 1] = c

        # Update the indices for the next interval
        start_idx = end_idx
        end_idx += interval_size
        if i == num_intervals - 2:  # Last interval may have more points
            end_idx = len(data)

    return intervals
