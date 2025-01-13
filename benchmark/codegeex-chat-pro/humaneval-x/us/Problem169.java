public class Problem169 {
    public static void main(String[] args) {
        Solution solution = new Solution();
        assert solution.numDistinct("rabbbit", "rabbit") == 3 : "Test 1 Failed";
        assert solution.numDistinct("babgbag", "bag") == 5 : "Test 2 Failed";
        assert solution.numDistinct("", "") == 1 : "Test 3 Failed";
        assert solution.numDistinct("a", "a") == 1 : "Test 4 Failed";
        assert solution.numDistinct("a", "b") == 0 : "Test 5 Failed";
        System.out.println("All tests passed!");
    }
}


class Solution {
    public int numDistinct(String s, String t) {
        int m = s.length(), n = t.length();
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 0; i <= m; i++) {
            dp[i][n] = 1;
        }

        for (int sIndex = m - 1; sIndex >= 0; sIndex--) {
            for (int tIndex = n - 1; tIndex >= 0; tIndex--) {
                dp[sIndex][tIndex] = dp[sIndex + 1][tIndex];

                if (s.charAt(sIndex) == t.charAt(tIndex)) {
                    dp[sIndex][tIndex] += dp[sIndex + 1][tIndex + 1];
                }
            }
        }

        return dp[0][0];
    }
}
