import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

class Solution {
    /**
     * Write 'Hello world' to the specified file.
     * >>> writeHelloWorld('hello_world.txt')
     */
    public void writeHelloWorld(String filename) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("Hello world");
        }
    }
}

public class HumanEvalJavaExtended164 {
    public static void main(String[] args) throws IOException {
        Solution s = new Solution();
        s.writeHelloWorld("hello_world.txt");
        BufferedReader reader = new BufferedReader(new FileReader("hello_world.txt"));
        String content = reader.readLine();
        reader.close();
        if (!content.equals("Hello world")) {
            throw new AssertionError();
        }
    }
}
