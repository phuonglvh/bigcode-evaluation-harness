
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

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
        JFrame window = new JFrame();
        window.setTitle("Login Form");
        window.setLayout(new GridLayout(3, 2));

        JLabel usernameLabel = new JLabel("Username:");
        window.add(usernameLabel);
        JTextField usernameEntry = new JTextField();
        window.add(usernameEntry);

        JLabel passwordLabel = new JLabel("Password:");
        window.add(passwordLabel);
        JPasswordField passwordEntry = new JPasswordField();
        window.add(passwordEntry);

        JButton submitButton = new JButton("Submit");
        submitButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                System.out.println("Submitted!");
            }
        });
        window.add(submitButton);

        Timer timer = new Timer(10000, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                window.dispose();
            }
        });
        timer.setRepeats(false);
        timer.start();

        window.pack();
        window.setVisible(true);
    }
}

public class Problem168 {

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
