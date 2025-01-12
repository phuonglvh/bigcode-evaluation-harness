from concurrent.futures import ThreadPoolExecutor


def parallel_sum_of_squares(n: int, num_threads: int) -> int:
    """
    Compute the sum of squares of numbers from 1 to n using parallel computation.
    >>> parallel_sum_of_squares(5, 2)
    55
    """
    def sum_of_squares(start: int, end: int) -> int:
        """Compute the sum of squares for numbers in the range [start, end)."""
        return sum(x * x for x in range(start, end))

    if n <= 0:
        return 0

    step = (n + num_threads - 1) // num_threads
    ranges = [(i, min(i + step, n + 1)) for i in range(1, n + 1, step)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda r: sum_of_squares(*r), ranges)

    return sum(results)


def check(parallel_sum_of_squares):
    # Test small numbers
    assert parallel_sum_of_squares(5, 2) == 55
    assert parallel_sum_of_squares(10, 2) == 385

    # Test edge cases
    assert parallel_sum_of_squares(0, 4) == 0  # No numbers to sum
    assert parallel_sum_of_squares(-10, 4) == 0  # Negative range

    # Test with larger n
    assert parallel_sum_of_squares(100, 4) == sum(x * x for x in range(1, 101))
    assert parallel_sum_of_squares(1000, 8) == sum(
        x * x for x in range(1, 1001))


check(parallel_sum_of_squares)
