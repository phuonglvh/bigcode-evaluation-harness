class Solution {
    /**
    brackets is a string of \"(\" and \")\".
    return True if every opening bracket has a corresponding closing bracket.
    
    >>> correctBracketing(\"(\")
    false
    >>> correctBracketing(\"()\")
    true
    >>> correctBracketing(\"(()())\")
    true
    >>> correctBracketing(\")(()\")
    false
    */
    public boolean correctBracketing(String brackets) {

        int balance = 0;
        for (char ch : brackets.toCharArray()) {
            if (ch == '(') {
                balance++;
            } else if (ch == ')') {
                balance--;
                if (balance < 0) {
                    return false;
                }
            }
        }
        return balance == 0;
    }
}