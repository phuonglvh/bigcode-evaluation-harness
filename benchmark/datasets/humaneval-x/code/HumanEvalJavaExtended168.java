import java.awt.GridLayout;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import javax.swing.Timer;

class Solution {
    /**
     * Create a simple login form with fields for username and password, and a
     * submit button.
     * When the submit button is pressed, 'Submitted!' is printed to the console.
     */
    public void createForm() {
        JFrame window = new JFrame("Login Form");
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setSize(300, 200);
        window.setLayout(new GridLayout(3, 2));

        JLabel usernameLabel = new JLabel("Username:");
        JTextField usernameEntry = new JTextField();
        JLabel passwordLabel = new JLabel("Password:");
        JPasswordField passwordEntry = new JPasswordField();
        JButton submitButton = new JButton("Submit");

        submitButton.addActionListener(e -> System.out.println("Submitted!"));

        window.add(usernameLabel);
        window.add(usernameEntry);
        window.add(passwordLabel);
        window.add(passwordEntry);
        window.add(submitButton);

        window.setVisible(true);

        // auto close after xs
        Timer timer = new Timer(10000, e -> window.dispose());
        timer.setRepeats(false);
        timer.start();
    }
}

public class HumanEvalJavaExtended168 {
    // Test if the form can be created
    public static void testCreateForm() {
        try {
            Solution solution = new Solution();
            solution.createForm();
            assert true; // If no exception is thrown, the form is created successfully
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            assert false;
        }
    }

    public static void main(String[] args) {
        testCreateForm();
    }
}
