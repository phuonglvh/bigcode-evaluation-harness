import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;

class Country {
    private JSONObject name;
    private boolean independent;

    public Country(JSONObject name, boolean independent) {
        this.name = name;
        this.independent = independent;
    }

    public JSONObject getName() {
        return name;
    }

    public boolean isIndependent() {
        return independent;
    }
}

class Solution {
    /**
     * Call the REST API at
     * https://restcountries.com/v3.1/independent?status=true&fields=name,independent
     * with GET
     * and return the list of independent countries.
     * Examples:
     * List<Country> countries = getIndependentCountries();
     */
    public List<Country> getIndependentCountries() throws Exception {
        String url = "https://restcountries.com/v3.1/independent?status=true&fields=name,independent";
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestMethod("GET");

        BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        StringBuilder response = new StringBuilder();
        String inputLine;
        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();

        JSONArray countriesData = new JSONArray(response.toString());
        List<Country> countries = new ArrayList<>();

        for (int i = 0; i < countriesData.length(); i++) {
            JSONObject countryData = countriesData.getJSONObject(i);
            JSONObject name = countryData.optJSONObject("name");
            boolean independent = countryData.optBoolean("independent", false);
            countries.add(new Country(name, independent));
        }

        return countries;
    }
}

public class HumanEvalJavaExtended165 {
    public static void main(String[] args) throws Exception {
        Solution solution = new Solution();
        List<Country> countries = solution.getIndependentCountries();

        assert countries != null;
        assert countries.size() > 0;
        assert countries.stream().allMatch(Country::isIndependent);
        assert countries.stream().allMatch(country -> country.getName() != null);
    }
}