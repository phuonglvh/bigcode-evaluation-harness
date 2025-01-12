import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

class Solution {
    /**
     * Connect to a PostgreSQL database with the given parameters: host, port,
     * username, password, and database.
     * Examples:
     * Connection conn = connectPostgres("localhost", 5432, "postgres", "password",
     * "postgres");
     */
    public Connection connectPostgres(String host, int port, String username, String password, String database)
            throws SQLException {
        String url = String.format("jdbc:postgresql://%s:%d/%s", host, port, database);
        Connection conn = DriverManager.getConnection(url, username, password);
        return conn;
    }
}

public class HumanEvalJavaExtended167 {
    public static void main(String[] args) throws SQLException {
        Solution solution = new Solution();

        // Test connection parameters
        Connection conn = solution.connectPostgres("localhost", 5432, "postgres", "password", "postgres");
        assert conn != null;

        // Test connection by executing a simple query
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery("SELECT 1");
        assert rs.next() && rs.getInt(1) == 1;
        rs.close();
        stmt.close();
        conn.close();
    }
}
