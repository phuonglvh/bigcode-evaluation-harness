# https://leetcode.com/problems/distinct-subsequences/solutions/4177156/optimizing-string-subsequence-counting-a-deep-dive-into-all-3-approaches-mr-robot/
# HARD leetcode problem

def numDistinct(s: str, t: str) -> int:
    """
    Given two strings s and t, return the number of distinct subsequences of s which equals t.

    A subsequence is a sequence that can be derived from another sequence by deleting some or no elements 
    without changing the order of the remaining elements.

    The function should return the number of distinct subsequences of s which equals t.

    Example 1:
    Input: s = "rabbbit", t = "rabbit"
    Output: 3

    Example 2:
    Input: s = "babgbag", t = "bag"
    Output: 5

    Constraints:
    - 1 <= s.length, t.length <= 1000
    - s and t consist of English letters (lowercase and uppercase).
    """
    def recursiveWithMemoization(s: str, t: str, s_ind: int, t_ind: int, dp) -> int:
        if t_ind == len(t):
            return 1
        if s_ind == len(s):
            return 0

        if dp[s_ind][t_ind] != -1:
            return dp[s_ind][t_ind]

        take, notTake = 0, 0

        if s[s_ind] == t[t_ind]:
            take = recursiveWithMemoization(s, t, s_ind + 1, t_ind + 1, dp)

        notTake = recursiveWithMemoization(s, t, s_ind + 1, t_ind, dp)

        dp[s_ind][t_ind] = take + notTake
        return dp[s_ind][t_ind]

    dp = [[-1 for _ in range(len(t))] for _ in range(len(s))]
    return recursiveWithMemoization(s, t, 0, 0, dp)

    
def check(numDistinct):
    assert numDistinct("rabbbit", "rabbit") == 3
    assert numDistinct("babgbag", "bag") == 5
    assert numDistinct("", "") == 1
    assert numDistinct("a", "a") == 1
    assert numDistinct("a", "b") == 0


check(numDistinct)
