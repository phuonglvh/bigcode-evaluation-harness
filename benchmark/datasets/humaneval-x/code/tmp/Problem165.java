
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

public class Problem165 {
    public List<Country> getIndependentCountries() throws Exception {
        String url = "https://restcountries.com/v3.1/independent?status=true&fields=name,independent";
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        connection.setRequestMethod("GET");

        BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        String line;
        StringBuilder responseContent = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            responseContent.append(line);
        }
        reader.close();
        connection.disconnect();

        JSONArray countriesData = new JSONArray(responseContent.toString());
        List<Country> countries = new ArrayList<>();
        for (int i = 0; i < countriesData.length(); i++) {
            JSONObject countryData = countriesData.getJSONObject(i);
            JSONObject name = countryData.getJSONObject("name");
            boolean independent = countryData.getBoolean("independent");
            countries.add(new Country(name, independent));
        }

        return countries;

    }

    public static void main(String[] args) throws Exception {
        Problem165 solution = new Problem165();
        List<Country> countries = solution.getIndependentCountries();

        assert countries != null;
        assert countries.size() > 0;
        assert countries.stream().allMatch(Country::isIndependent);
        assert countries.stream().allMatch(country -> country.getName() != null);
    }
}