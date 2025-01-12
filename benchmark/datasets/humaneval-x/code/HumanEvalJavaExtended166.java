// import java.util.ArrayList;
// import java.util.Arrays;
// import java.util.List;
// import java.util.concurrent.Callable;
// import java.util.concurrent.ExecutorService;
// import java.util.concurrent.Executors;
// import java.util.concurrent.Future;
// import java.util.stream.IntStream;

// class Solution {
//         /**
//          * Compute the sum of squares of numbers from 1 to n using parallel computation.
//          * Examples:
//          * parallelSumOfSquares(5, 2) -> 55
//          */
//         public int parallelSumOfSquares(int n, int numThreads) {
//         if (n <= 0) {
//             return 0;
//         }

//         int step = (n + numThreads - 1) / numThreads;
//         List<int[]> ranges = new ArrayList<>();
//         for (int i = 1; i <= n; i += step) {
//             ranges.add(new int[] { i, Math.min(i + step, n + 1) });
//         }

//         ExecutorService executor = Executors.newFixedThreadPool(numThreads);
//         List<Callable<Integer>> tasks = new ArrayList<>();

//         for (int[] range : ranges) {
//             tasks.add(() -> sumOfSquares(range[0], range[1]));
//         }

//         int total = 0;
//         try {
//             List<Future<Integer>> results = executor.invokeAll(tasks);
//             for (Future<Integer> result : results) {
//                 total += result.get();
//             }
//         } catch (Exception e) {
//             e.printStackTrace();
//         } finally {
//             executor.shutdown();
//         }

//         return total;
//     }

//     private int sumOfSquares(int start, int end) {
//         int sum = 0;
//         for (int i = start; i < end; i++) {
//             sum += i * i;
//         }
//         return sum;
//     }
// }

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

class Solution {
    /**
     * Compute the sum of squares of numbers from 1 to n using parallel computation.
     * Examples:
     * parallelSumOfSquares(5, 2) -> 55
     */
    public int parallelSumOfSquares(int n, int numThreads) {
        if (n <= 0) {
            return 0;
        }

        int step = (n + numThreads - 1) / numThreads;
        List<int[]> ranges = new ArrayList<>();
        for (int i = 1; i <= n; i += step) {
            ranges.add(new int[] { i, Math.min(i + step, n + 1) });
        }

        ForkJoinPool forkJoinPool = new ForkJoinPool(numThreads);
        List<Callable<Integer>> tasks = new ArrayList<>();

        for (int[] range : ranges) {
            tasks.add(() -> sumOfSquares(range[0], range[1]));
        }

        int total = 0;
        try {
            List<Future<Integer>> results = forkJoinPool.invokeAll(tasks);
            for (Future<Integer> result : results) {
                total += result.get();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            forkJoinPool.shutdown();
        }

        return total;
    }

    private int sumOfSquares(int start, int end) {
        int sum = 0;
        for (int i = start; i < end; i++) {
            sum += i * i;
        }
        return sum;
    }
}

public class HumanEvalJavaExtended166 {
    public static void main(String[] args) {
        Solution solution = new Solution();
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