
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

public class Problem166 {
    public int parallelSumOfSquares(int n, int numThreads) {
        if (n <= 0) {
            return 0;
        }

        int step = (n + numThreads - 1) / numThreads;
        List<Callable<Integer>> tasks = new ArrayList<>();

        for (int i = 1; i <= n; i += step) {
            int start = i;
            int end = Math.min(i + step, n + 1);
            tasks.add(() -> sumOfSquares(start, end));
        }

        ForkJoinPool pool = new ForkJoinPool(numThreads);
        List<Future<Integer>> futures = pool.invokeAll(tasks);

        int sum = 0;
        for (Future<Integer> future : futures) {
            try {
                sum += future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        pool.shutdown();
        return sum;
    }

    private int sumOfSquares(int start, int end) {
        int sum = 0;
        for (int i = start; i < end; i++) {
            sum += i * i;
        }
        return sum;

    }

    public static void main(String[] args) {
        Problem166 solution = new Problem166();
        List<Boolean> tests = Arrays.asList(
                solution.parallelSumOfSquares(5, 2) == 55,
                solution.parallelSumOfSquares(10, 2) == 385,
                solution.parallelSumOfSquares(0, 4) == 0,
                solution.parallelSumOfSquares(-10, 4) == 0,
                solution.parallelSumOfSquares(100, 4) == IntStream.rangeClosed(1, 100).map(x -> x * x).sum());

        if (tests.contains(false)) {
            throw new AssertionError("Test failed");
        }
    }
}