import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class Problem167 {
        public Connection connectPostgres(String host, int port, String username, String password, String database) throws SQLException {
        String url = "jdbc:postgresql://" + host + ":" + port + "/" + database;
        return DriverManager.getConnection(url, username, password);

    }
    public static void main(String[] args) throws SQLException {
        Problem167 solution = new Problem167();

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