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
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][n] = 1

    for sIndex in range(m - 1, -1, -1):
        for tIndex in range(n - 1, -1, -1):
            dp[sIndex][tIndex] = dp[sIndex + 1][tIndex]

            if s[sIndex] == t[tIndex]:
                dp[sIndex][tIndex] += dp[sIndex + 1][tIndex + 1]

    return dp[0][0]


def check(numDistinct):
    assert numDistinct("rabbbit", "rabbit") == 3
    assert numDistinct("babgbag", "bag") == 5
    assert numDistinct("", "") == 1
    assert numDistinct("a", "a") == 1
    assert numDistinct("a", "b") == 0


check(numDistinct)
